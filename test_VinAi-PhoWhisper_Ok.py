
from transformers import pipeline

path_model = "vinai/PhoWhisper-small"

audio_file = "tin-tuc.mp3"

transcriber = pipeline("automatic-speech-recognition", model=path_model)
output = transcriber(audio_file)['text'] # path_to_audio_with_sampling_rate_16kHz

print(output)

"""
# audio_file = "tin-tuc.mp3"
% python test_VinAi-PhoWhisper_Ok.py

Due to a bug fix in https://github.com/huggingface/transformers/pull/28687 transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English.This might be a breaking change for your use case. If you want to instead always translate your audio to English, make sure to pass `language='en'`.

GS.TSKH Nguyễn Mại phân tích cơ hội đưa Việt Nam trở thành cường quốc trong lĩnh vực 'nóng' nhất toàn cầu và khả năng tăng trưởng GDP vượt mức 7% năm 2024, giáo sư tiến sĩ khoa học nguyễn mại phân tích cơ hội đưa việt nam trở thành cường quốc trong lĩnh vực nóng nhất toàn cầu và khả năng tăng trưởng gdp vượt bước bảy phần trăm năm hai nghìn không trăm hai mươi bốn. Từ năm 2023 đến nay, Việt Nam đã đón gần 30 chuyến thăm của lãnh đạo cấp cao các nước, trong đó có 2 chuyến thăm lịch sử của Mỹ và Trung Quốc chỉ cách nhau hơn 1 tháng. Cùng với các chuyến thăm đó là hàng loạt đoàn doanh nghiệp, quỹ quốc tế đến tìm hiểu các cơ hội đầu tư, kinh doanh. Theo GS. TSKH Nguyễn Mại, đây là những tín hiệu tích cực để có thể dự báo rằng Việt Nam sẽ trở thành điểm đến lý tưởng cho dòng vốn FDI mới.

"""