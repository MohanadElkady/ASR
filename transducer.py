import nemo.collections.asr as nemo_asr
from nemo.core.classes import ModelPT, Exportable
#model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_large")
model=nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_conformer_ctc_large")
#model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_citrinet_256_ls")
import glob
import json
import os
import tempfile
from argparse import ArgumentParser
import torch
from tqdm import tqdm
import sys
import os
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import ONNXGreedyBatchedRNNTInfer
from nemo.utils import logging
import onnxoptimizer
import onnx




def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "--nemo_model", type=str, default=None, required=False, help="Path to .nemo file",
    )
    parser.add_argument(
        '--pretrained_model', type=str, default=None, required=False, help='Name of a pretrained NeMo file'
    )
    parser.add_argument('--onnx_encoder', type=str, default=None, required=False, help="Path to onnx encoder model")
    parser.add_argument(
        '--onnx_decoder', type=str, default=None, required=False, help="Path to onnx decoder + joint model"
    )
    parser.add_argument('--threshold', type=float, default=0.01, required=False)

    parser.add_argument('--audio_dir', type=str, default=None, required=False, help='Path to the test audio file')
    parser.add_argument('--audio_type', type=str, default='wav', help='File format of audio')

    parser.add_argument('--export', action='store_true', help="Whether to export the model into onnx prior to eval")
    parser.add_argument('--max_symbold_per_step', type=int, default=5, required=False, help='Number of decoding steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batchsize')
    parser.add_argument('--log', action='store_true', help='Log the predictions between pytorch and onnx')

    args = parser.parse_args()
    return args



def assert_args(args):
    if args.nemo_model is None and args.pretrained_model is None:
        raise ValueError(
            "`nemo_model` or `pretrained_model` must be passed! It is required for decoding the RNNT tokens "
            "and ensuring predictions match between Torch and ONNX."
        )

    if args.nemo_model is not None and args.pretrained_model is not None:
        raise ValueError(
            "`nemo_model` and `pretrained_model` cannot both be passed! Only one can be passed to this script."
        )

    if args.export and (args.onnx_encoder is not None or args.onnx_decoder is not None):
        raise ValueError("If `export` is set, then `onnx_encoder` and `onnx_decoder` arguments must be None")

    if args.audio_file is None:
        raise ValueError("Please provide the path to the test audio file using the `--audio_file` argument.")

    if int(args.max_symbold_per_step) < 1:
        raise ValueError("`max_symbold_per_step` must be an integer > 0")


def export_model_if_required(args, nemo_model):
    if args.export:
        nemo_model.export("temp_rnnt.onnx")
        args.onnx_encoder = "encoder-temp_rnnt.onnx"
        args.onnx_decoder = "decoder_joint-temp_rnnt.onnx"


def resolve_audio_filepaths(args):
    # get audio filenames
    if args.audio_dir is not None:
        filepaths = list(glob.glob(os.path.join(args.audio_dir.audio_dir, f"*.{args.audio_type}")))
    else:
        # get filenames from manifest
        filepaths = []
        with open(args.dataset_manifest, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                filepaths.append(item['audio_filepath'])

    logging.info(f"\nTranscribing {len(filepaths)} files...\n")

    return filepaths

def main():
    args = parse_arguments()

    device = 'cpu'

    # Instantiate pytorch model
    if args.nemo_model is not None:
        nemo_model = args.nemo_model
        nemo_model = ASRModel.restore_from(nemo_model, map_location=device)  # type: ASRModel
        nemo_model.freeze()
    elif args.pretrained_model is not None:
        nemo_model = args.pretrained_model
        nemo_model = ASRModel.from_pretrained(nemo_model, map_location=device)  # type: ASRModel
        nemo_model.freeze()
    else:
        raise ValueError("Please pass either `nemo_model` or `pretrained_model`!")

    if torch.cuda.is_available():
        nemo_model = nemo_model.to('cuda')

    export_model_if_required(args, nemo_model)

    # Instantiate RNNT Decoding loop
    encoder_model = args.onnx_encoder
    decoder_model = args.onnx_decoder
    max_symbols_per_step = args.max_symbold_per_step
    decoding = ONNXGreedyBatchedRNNTInfer(encoder_model, decoder_model, max_symbols_per_step)

    audio_filepath = "2086-149220-0033.wav"

    # Evaluate Pytorch Model (CPU/GPU)
    actual_transcripts = nemo_model.transcribe([audio_filepath], batch_size=args.batch_size)[0]
    

    # Evaluate ONNX model
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
            entry = {'audio_filepath': audio_filepath, 'duration': 100000, 'text': 'nothing'}
            fp.write(json.dumps(entry) + '\n')

        config = {'paths2audio_files': [audio_filepath], 'batch_size': args.batch_size, 'temp_dir': tmpdir}

        nemo_model.preprocessor.featurizer.dither = 0.0
        nemo_model.preprocessor.featurizer.pad_to = 0

        temporary_datalayer = nemo_model._setup_transcribe_dataloader(config)

        all_hypothesis = []
        
        for test_batch in tqdm(temporary_datalayer, desc="ONNX Transcribing"):
            input_signal, input_signal_length = test_batch[0], test_batch[1]
            input_signal = input_signal.to('cuda:0')
            input_signal_length = input_signal_length.to('cuda:0')

            # Acoustic features
            processed_audio, processed_audio_len = nemo_model.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )
            # RNNT Decoding loop
            hypotheses = decoding(audio_signal=processed_audio, length=processed_audio_len)

            # Process hypothesis (map char/subword token ids to text)
            hypotheses = nemo_model.decoding.decode_hypothesis(hypotheses)  # type: List[str]
            
            for hyp in hypotheses:
                hypothesis_text = hyp.text
                all_hypothesis.append(hypothesis_text)

            # Extract text from the hypothesis
            texts = [h.text for h in hypotheses]

            all_hypothesis += texts
            del processed_audio, processed_audio_len
            del test_batch
    	

    if args.log:
        for pt_transcript, onnx_transcript in zip(actual_transcripts, all_hypothesis):
            print(f"Pytorch Transcripts : {pt_transcript}")
            print(f"ONNX Transcripts    : {onnx_transcript}")
        print()

    # Measure error rate between onnx and pytorch transcipts
    print(all_hypothesis)
    print(actual_transcripts)
    pt_onnx_cer = word_error_rate(all_hypothesis, actual_transcripts, use_cer=True)
    assert pt_onnx_cer < args.threshold, "Threshold violation!"

    print("Character error rate between Pytorch and ONNX:", pt_onnx_cer)


if __name__ == '__main__':
    main()
