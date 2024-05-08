# test_VinAi-PhoWhisper

from transformers import pipeline <br><br>

path_model = "vinai/PhoWhisper-small" <br><br>

audio_file = "tin-tuc.mp3" <br><br>

transcriber = pipeline("automatic-speech-recognition", model=path_model)<br>
output = transcriber(audio_file)['text'] # path_to_audio_with_sampling_rate_16kHz<br><br>

print(output)
