from pydub import AudioSegment

# WAVファイルを読み込み
try:
    sound = AudioSegment.from_wav("silent.wav")
    # MP3ファイルとして書き出し
    sound.export("silent.mp3", format="mp3")
    print("MP3ファイル 'silent.mp3' を生成しました。")
except FileNotFoundError:
    print("エラー: 'silent.wav' が見つかりません。先にWAVファイルを生成してください。")
