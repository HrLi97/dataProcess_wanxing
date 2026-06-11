# pip install silero-vad

# import torch
# import torchaudio
import os
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

def process_audio_file(input_path, output_dir="output_clips"):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在加载模型...")
    # 1. 加载模型 (首次运行会自动下载模型文件)
    model = load_silero_vad()

    print(f"正在读取音频: {input_path}")
    # 2. 读取音频
    # 注意：read_audio 会自动将音频重采样为 16kHz 单声道，这是 Silero VAD 要求的格式
    wav = read_audio(input_path) 
    
    # 获取音频的采样率 (read_audio 默认返回 16000)
    sampling_rate = 16000 

    print("正在进行语音活动检测 (VAD)...")
    # 3. 获取语音时间戳
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sampling_rate,
        threshold=0.5,                  # 语音概率阈值，越高越严格
        min_speech_duration_ms=250,     # 小于 250ms 的语音会被忽略（过滤极短杂音）
        max_speech_duration_s=float('inf'), # 最大语音时长限制
        min_silence_duration_ms=100,    # 语音中间小于 100ms 的静音会被合并（防止一句话被切碎）
        speech_pad_ms=30,               # 在检测到的语音前后各增加 30ms 的 padding，防止首尾被截断
        return_seconds=True             # 返回秒数 (False 则返回采样点索引)
    )

    print(f"检测完成！共发现 {len(speech_timestamps)} 个语音片段。")

    # 4. 裁剪并保存语音片段
    for i, timestamp in enumerate(speech_timestamps):
        start_sec = timestamp['start']
        end_sec = timestamp['end']
        
        # 将秒数转换回采样点索引，用于在 tensor 中切片
        start_sample = int(start_sec * sampling_rate)
        end_sample = int(end_sec * sampling_rate)
        
        # 裁剪音频片段 (wav 是一个 tensor)
        speech_chunk = wav[start_sample:end_sample]
        
        # 保存为新的 wav 文件
        output_path = os.path.join(output_dir, f"speech_clip_{i+1}.wav")
        # torchaudio.save(output_path, speech_chunk.unsqueeze(0), sampling_rate)
        print(f"已保存片段 {i+1}: {start_sec:.2f}s - {end_sec:.2f}s -> {output_path}")

if __name__ == "__main__":
    # 请将 'your_audio.wav' 替换为你实际的音频文件路径
    # 建议准备一段包含说话和静音的 wav 文件进行测试
    audio_file = 'audios_nomusic'
    audio_list = os.listdir(audio_file)
    for audio in audio_list:
        if audio.endswith('.mp3'):
            print(f"正在处理文件: {audio}")
            process_audio_file(os.path.join(audio_file, audio))
    # INPUT_AUDIO = 'your_audio.wav' 
    # process_audio_file(INPUT_AUDIO)