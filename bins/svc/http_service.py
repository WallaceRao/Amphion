from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import multiprocessing
import os
import json
import base64
import logging
from datetime import datetime
import random
import numpy as np
from service_inference import process_request

svc_logger = logging.getLogger("svc_service")
supported_singers = {"vocalist_l1_王菲", 
                     "vocalist_l1_张学友",
                     "vocalist_l1_李健",
                     "vocalist_l1_汪峰",
                     "vocalist_l1_石倚洁",
                     "vocalist_l1_蔡琴",
                     "vocalist_l1_那英",
                     "vocalist_l1_陈奕迅",
                     "vocalist_l1_陶喆",
                     "vocalist_l1_Adele",
                     "vocalist_l1_Beyonce",
                     "vocalist_l1_BrunoMars",
                     "vocalist_l1_JohnMayer",
                     "vocalist_l1_MichaelJackson",
                     "vocalist_l1_TaylorSwift"}

supported_data_format = {"mp3", "wav"}

def get_request_work_folder(work_dir):
    now = datetime.now()
    dt_string = now.strftime("%Y%d%m_%H%M%S")
    return work_dir + "/" + dt_string + "_" + str(random.randint(1, 1000))

class Handler(BaseHTTPRequestHandler):

    def send_post_response(self, response_str):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(len(response_str.encode("UTF-8"))))
        self.end_headers()
        self.wfile.write(response_str.encode("UTF-8"))

    def do_POST(self):
        data_string = self.rfile.read(int(self.headers['Content-Length']))
        json_obj = None
        try:
            json_obj = json.loads(data_string)
        except Exception as e:
            svc_logger.info(f"could not parse request:{data_string}")
        err_msg = ""
        if "singer_name" not in json_obj.keys():
            err_msg = "no singer name provided"
        elif json_obj["singer_name"] not in supported_singers:
            err_msg = "unkown singer:" + json_obj["singer_name"]
        if "data_format" not in json_obj.keys():
            err_msg = "no data format provided"
        elif json_obj["data_format"] not in supported_data_format:
            err_msg = "only mp3 and wav are supported, unkown format:" + json_obj["data_format"]
        if "audio_data" not in json_obj.keys():
            err_msg = "no data provided"
        if err_msg != "":
            response_str = json.dumps({"err_msg": err_msg})
            self.send_post_response(response_str)
            return
        singer_name = json_obj["singer_name"]
        format = json_obj["data_format"]
        audio_data = json_obj["audio_data"]
        audio_binary = base64.b64decode(audio_data)
        pitch_np = None
        if "pitch_data" in json_obj.keys():
            pitch_data = json_obj["pitch_data"]
            pitch_binary = base64.b64decode(pitch_data)
            pitch_np = np.frombuffer(pitch_binary, dtype=np.float64)
        action = "all"
        if "action" in json_obj.keys():
            action = json_obj["action"]
        work_folder = os.path.abspath("./work_dir")
        request_work_folder = get_request_work_folder(work_folder)
        os.makedirs(request_work_folder, exist_ok=True)
        os.makedirs(request_work_folder + "/input_wav", exist_ok=True)
        os.makedirs(request_work_folder + "/output_wav", exist_ok=True)
        audio_file_path = request_work_folder + "/input_wav/input.wav"
        if format == "mp3":
            audio_file_path = request_work_folder + "/input_wav/input.mp3"
        with open(audio_file_path, 'wb+') as file:
            file.write(audio_binary)
        if action == "pitch_only":
            f0_bytes, audio_bytes = process_request(request_work_folder, action, singer_name, pitch_np, format)
            if f0_bytes is None:
                err_msg = "process finished but got no result, unknown reason."
                response_str = json.dumps({"err_msg": err_msg})
                self.send_post_response(response_str)
                return
            f0_base64_str = base64.b64encode(f0_bytes)
            response_str = json.dumps({"sample_rate": "24000",
                                       "pitch_data": f0_base64_str.decode(),
                                       "err_msg":err_msg})
        else:
            f0_bytes, audio_bytes = process_request(request_work_folder, action, singer_name, pitch_np, format)
            if audio_bytes is None:
                err_msg = "process finished but got no result, unknown reason."
                response_str = json.dumps({"err_msg": err_msg})
                self.send_post_response(response_str)
                return
            audio_base64_str = base64.b64encode(audio_bytes)
            response_str = json.dumps({"sample_rate": "24000", "audio_data": audio_base64_str.decode(), "err_msg":err_msg})
        self.send_post_response(response_str)

def run():
    port = 80
    server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    svc_logger.info(f"server started on port: {port}")
    server.serve_forever()

if __name__ == '__main__':
    os.environ["WORK_DIR"] = "./"
    run()