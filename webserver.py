import argparse
import asyncio
import json
import logging
import os
import shutil
import traceback
from datetime import datetime
import shutil

from convert import convert_and_save_brainflow_file
from run_yasa import load_mne_fif_and_run_yasa
from upload import upload_dir_to_gcs, upload_file_to_gcs
from websocket import WebsocketHandler
import run_feature_pipeline

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# "08-07-2024--22-51-16.brainflow.csv" -> "08-07-2024--22-51-16"
def output_dirname(filename: str) -> str:
    input_file_without_ext = os.path.splitext(filename)[0].replace(".brainflow", "")
    return input_file_without_ext

async def run_webserver():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, required=True, help='The Brainflow board ID to connect to')
    parser.add_argument('-wp', '--websocket_port', default=9090, type=int, help='Websocket port')
    parser.add_argument('--ssl_cert', type=str, help='SSL cert file for websocket server')
    parser.add_argument('--ssl_key', type=str, help='SSL key file for websocket server')
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)

    logger.info(f"Starting Brainwave Processor with args: {args}")

    done = False

    def log(msg):
        logger.info(msg)
        print(msg)
        asyncio.create_task(websocket_handler.broadcast_websocket_message(json.dumps({
            'address': 'log',
            'msg': msg
        })))


    # "08-07-2024--22-51-16" -> "/path/to/08-07-2024--22-51-16"
    def output_dir(filename: str) -> str:
        out = str(os.path.join(args.data_dir, output_dirname(filename)))
        os.makedirs(out, exist_ok=True)
        return out

    def on_websocket_message(msg):
        nonlocal done
        logging.info(f"Command received: {msg}")

        if msg['command'] == 'files':
            log('Files command received')
            files_info = []
            for file in os.listdir(args.data_dir):
                file_path = os.path.join(args.data_dir, file)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                elif os.path.isdir(file_path):
                    file_size = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(file_path) for filename in filenames)
                else:
                    file_size = 0  # Or handle the case where the path is neither a file nor a directory
                isfile = os.path.isfile(file_path)
                lastmodified = os.path.getmtime(file_path)
                files_info.append({'name': file, 'size': file_size, 'isfile': isfile, 'lastmodified': lastmodified})
            try:
                diskusage_info = shutil.disk_usage(args.data_dir)
                diskusage = diskusage_info.total
                diskremaining = diskusage_info.free
            except Exception as e:
                log("Error getting disk usage " + str(e))
                diskusage = 0
                diskremaining = 0
            asyncio.create_task(websocket_handler.broadcast_websocket_message(json.dumps({
                'address': 'files',
                'diskusage': diskusage,
                'diskremaining': diskremaining,
                'data': files_info
            })))

        elif msg['command'] == 'do_it_all':
            data = msg['data']
            filename = data['file']
            channels = data['channels']

            full_input_filename = str(os.path.join(args.data_dir, filename))
            od = output_dirname(filename)
            full_output_dirname = output_dir(filename)
            full_output_filename = str(os.path.join(full_output_dirname, 'raw.fif'))

            log(f'Input {full_input_filename} to output dir {full_output_dirname} with channels {channels}')

            convert_and_save_brainflow_file(log, full_input_filename, full_output_filename, channels)
            upload_file_to_gcs(log, 'examined-life-input-eeg-raw', full_output_filename, od)
            load_mne_fif_and_run_yasa(log, full_output_filename)
            upload_dir_to_gcs(log, 'examined-life-derived-eeg', full_output_dirname, od)

        elif msg['command'] == 'pipeline':
            data = msg['data']
            filename = data['file']
            channels = data['channels']

            full_input_filename = str(os.path.join(args.data_dir, filename))
            od = output_dirname(filename)
            full_output_dirname = output_dir(filename)
            full_output_filename = str(os.path.join(full_output_dirname, 'raw.fif'))

            log(f'Input {full_input_filename} to output dir {full_output_dirname} with channels {channels}')

            convert_and_save_brainflow_file(log, full_input_filename, full_output_filename, channels)
            upload_file_to_gcs(log, 'examined-life-input-eeg-raw', full_output_filename, od)
            run_feature_pipeline.pipeline(log, full_output_filename)
            upload_dir_to_gcs(log, 'examined-life-derived-eeg', full_output_dirname, od)

        # Saving EDF is expensive and generally gets killed, so we use FIF
        elif msg['command'] == 'convert_to_fif':
            data = msg['data']
            filename = data['file']
            channels = data['channels']
            full_input_filename = str(os.path.join(args.data_dir, filename))
            full_output_filename = str(os.path.join(output_dir(filename), 'raw.fif'))
            log(f'Converting {full_input_filename} to {full_output_filename} with channels {channels}')
            convert_and_save_brainflow_file(log, full_input_filename, full_output_filename, channels)

        elif msg['command'] == 'convert_to_edf':
            data = msg['data']
            filename = data['file']
            channels = data['channels']
            full_input_filename = str(os.path.join(args.data_dir, filename))
            full_output_filename = str(os.path.join(output_dir(filename), 'raw.edf'))
            log(f'Converting {full_input_filename} to {full_output_filename} with channels {channels}')
            convert_and_save_brainflow_file(log, full_input_filename, full_output_filename, channels)

        elif msg['command'] == 'process':
            data = msg['data']
            dir = data['dir']
            full_input_dirname = str(os.path.join(args.data_dir, dir))
            full_input_filename = os.path.join(full_input_dirname, 'raw.fif')
            load_mne_fif_and_run_yasa(log, full_input_filename)

        elif msg['command'] == 'folder_pipeline':
            data = msg['data']
            dir = data['dir']
            full_input_dirname = str(os.path.join(args.data_dir, dir))
            full_input_filename = os.path.join(full_input_dirname, 'raw.fif')
            run_feature_pipeline.pipeline(log, full_input_filename)
            upload_dir_to_gcs(log, 'examined-life-derived-eeg', full_input_dirname, dir)

        elif msg['command'] == 'upload':
            data = msg['data']
            dir = data['dir']
            full_input_dirname = str(os.path.join(args.data_dir, dir))
            upload_dir_to_gcs(log, 'examined-life-input-eeg-raw', full_input_dirname, dir)
            upload_dir_to_gcs(log, 'examined-life-derived-eeg', full_input_dirname, dir)

        elif msg['command'] == 'delete':
            data = msg['data']
            filename = data['file']
            full_input_filename = str(os.path.join(args.data_dir, filename))
            if os.path.isfile(full_input_filename):
                os.remove(full_input_filename)
            elif os.path.isdir(full_input_filename):
                shutil.rmtree(full_input_filename)
            else:
                log(f"Path does not exist: {full_input_filename}")

        elif msg['command'] == 'delete_all_small_files':
            for file in os.listdir(args.data_dir):
                file_path = os.path.join(args.data_dir, file)
                file_size = os.path.getsize(file_path)
                isfile = os.path.isfile(file_path)
                if isfile and file_size < 200_000_000:
                    log(f"Deleting small file (size {file_size}): {file_path}")
                    os.remove(file_path)

        elif msg['command'] == 'quit':
            done = True

        else:
            log('Unknown command')


    websocket_handler = WebsocketHandler(args.ssl_cert, args.ssl_key, on_websocket_message)

    websocket_server_task = None
    if args.websocket_port:
        logger.info("Starting websocket server")
        websocket_server_task = asyncio.create_task(websocket_handler.start_websocket_server(args.websocket_port))

    while not done:

        try:
            await asyncio.sleep(10 / 1000)
        except Exception as e:
            logger.error(f"Error: {e}")
            traceback.print_exc()
            pass

    logger.info('Done')


if __name__ == "__main__":
    asyncio.run(run_webserver())
