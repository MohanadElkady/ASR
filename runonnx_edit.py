import onnxruntime as ort
import time
import librosa
import kaldi_native_fbank as knf
import numpy as np
import mutagen
from mutagen.wave import WAVE

def compute_feat(filename):
    sample_rate = 16000
    samples, _ = librosa.load(filename, sr=sample_rate)
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(sample_rate, (samples * 32768).tolist())
    online_fbank.input_finished()

    features = np.stack(
        [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
    )
    assert features.data.contiguous is True
    assert features.dtype == np.float32, features.dtype
    mean = features.mean(axis=0, keepdims=True)
    stddev = features.std(axis=0, keepdims=True)
    features = (features - mean) / (stddev + 1e-5)
    return features
def audio_duration(length):
    hours = length // 3600  # calculate in hours
    length %= 3600
    mins = length // 60  # calculate in minutes
    length %= 60
    seconds = length  # calculate in seconds
    total_duration = hours * 3600 + mins * 60 + seconds
    return float(total_duration)


def main():
    onnxtime=0.0
    onnx_model_path = "citrinet/cit_512_rnnt.onnx"
    session = ort.InferenceSession(onnx_model_path)
    
    audios = ["sample1.wav","sample2.wav","sample3.wav"]
    res=1.04
    for i in range(len(audios)):
        filename = audios[i]
        #Create a wave object of the audio file
        audio = WAVE(filename)
        
        # contains all the metadata about the wavpack file
        audio_info = audio.info
        length = int(audio_info.length)
    
        #get features of the filename
        features = compute_feat(filename)  # (T, C)
        features = np.expand_dims(features, axis=0)  # (N, T, C)
        features = features.transpose(0, 2, 1)  # (N, C, T)
        features_length = np.array([features.shape[2]], dtype=np.int64)
        # evaluate the model
        start = time.time()
        inputs = {
            session.get_inputs()[0].name: features,
            session.get_inputs()[1].name: features_length,
        }
        outputs = session.run([session.get_outputs()[0].name], input_feed=inputs)
        end = time.time()
        onnxtime +=round(end - start, 4)
        
    print("ONNX Runtime inference time: ", onnxtime)
    rtf=onnxtime/res
    print("RTF =" ,rtf)
    
if __name__ == '__main__':
    main()     
