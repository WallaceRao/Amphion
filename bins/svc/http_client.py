import requests
import numpy as np
import json
import base64
import time
import datetime
import sys    
import http.client

if __name__ == "__main__":
    conn = http.client.HTTPConnection('127.0.0.1:8094')
    data_format = "wav" # only MP3 and pcm are supported 
    action = 'all'
    overwrite_f0_file = "./pitch.dat"
    overwrite_f0 = np.fromfile(overwrite_f0_file, dtype=np.float64)
    print("1 overwrite_f0:", overwrite_f0)
    for i in range(overwrite_f0.shape[0]):
        if overwrite_f0[i] != 0:
            overwrite_f0[i] = overwrite_f0[i] + 50
    print("2 overwrite_f0:", overwrite_f0)
    pitch_bytes = overwrite_f0.tobytes()
    request_headers = {'Content-type': 'application/json'}
    audio_bytes = None
    file = '/mnt/data2/share/raoyonghui/svc_service/Amphion/work_dir/20240202_121212/input_wav/long_original.wav'
    with open(file, "rb") as file:
        audio_bytes = file.read()
    base64_str = base64.b64encode(audio_bytes)
    base64_pitch_bytes = base64.b64encode(pitch_bytes)
    foo = {
        'audio_data': base64_str.decode(),
        'singer_name': 'vocalist_l1_王菲',
        'pitch_data': base64_pitch_bytes.decode(),
        'action': action,
        'data_format':data_format
    }
    #foo = {
    #    'audio_data': base64_str.decode(),
    #    'singer_name': 'vocalist_l1_王菲',
    #    'pitch_data': base64_pitch_bytes.decode(),
    #    'action': action,
    #    'data_format':data_format
    #}
    json_data = json.dumps(foo)
    start = time.time()
    res = conn.request("POST", "", json_data, request_headers)
    response = conn.getresponse()
    print(response.status, response.reason)
    response_bytes = response.read()
    response_headers = response.getheaders()
    json_obj = json.loads(response_bytes)
    err_msg = ""
    if "err_msg" in json_obj.keys():
        err_msg = json_obj["err_msg"]

    samples = None
    if action == 'pitch_only':
        if "pitch_data" not in json_obj.keys():
            print("no pitch_data field returned, error msg:", err_msg)
            sys.exit(1)
        data_str = json_obj["pitch_data"]
        decoded_binary = base64.b64decode(data_str)
        with open("./pitch.dat", "wb") as file:
            file.write(decoded_binary)
    else:
        if "audio_data" not in json_obj.keys():
            print("no audio_data field returned, error msg:", err_msg)
            sys.exit(1)
        data_str = json_obj["audio_data"]
        decoded_binary = base64.b64decode(data_str)
        if data_format == "wav":
            with open("./result.wav", "wb") as file:
                file.write(decoded_binary)
        else:
            print(f'Time Used: {time.time() - start}, received mp3 bytes: {len(decoded_binary)}')
            with open("./result.mp3", "wb") as file:
                file.write(decoded_binary)