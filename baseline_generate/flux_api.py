# Please go to https://bfl.ai/ to get your black forest labs api key

import os
import requests
import base64
from PIL import Image
from io import BytesIO
import pandas as pd
import ast
import PIL
import io
import time

os.environ['BFL_API_KEY'] = '<your black forest labs api key here>'
root_dir = '../image_generate/input_images_resize_512/'
csv_path = '../image_generate/oai_instruction_generation_output.csv'
df = pd.read_csv(csv_path)
df = df[df['turns'] == 3]
print(df.iloc[0])



for kk in range(len(df)):
    try:
        raw_record = df.iloc[kk]
        prompt = (raw_record['instructions'])
        prompt = ast.literal_eval(prompt)
        prompt = [promp.strip('.').strip('"').strip('\'') for promp in prompt]
        image_idx = raw_record['image_index']
        image_path = os.path.join(root_dir, str(image_idx)+'_input_raw.jpg')
        image = PIL.Image.open(image_path)
        buffered = BytesIO()
        image.save(buffered, format="JPEG") # Or "PNG" if your image is PNG
        img_str = base64.b64encode(buffered.getvalue()).decode()

        for turn in [1,2,3]:
            turns = str(turn)
            prompt_i = prompt[turn-1]
            print(prompt_i)
           
            if turn == 1:
                request = requests.post(
                    'https://api.bfl.ai/v1/flux-kontext-max',
                    headers={
                        'accept': 'application/json',
                        'x-key': os.environ.get("BFL_API_KEY"),
                        'Content-Type': 'application/json',
                    },
                    json={
                        'prompt': prompt_i,
                        'input_image': img_str,
                    },
                ).json()

                print(request)
                request_id = request["id"]
                polling_url = request["polling_url"] # Use this URL for polling
            else:
                previous_image_path = './flux/multipass/' + str(image_idx)+f'_input_raw_turn_{turn-1}.png'
                image = PIL.Image.open(previous_image_path)
                buffered = BytesIO()
                image.save(buffered, format="PNG") # Or "PNG" if your image is PNG
                img_str = base64.b64encode(buffered.getvalue()).decode()
                request = requests.post(
                    'https://api.bfl.ai/v1/flux-kontext-max',
                    headers={
                        'accept': 'application/json',
                        'x-key': os.environ.get("BFL_API_KEY"),
                        'Content-Type': 'application/json',
                    },
                    json={
                        'prompt': prompt_i,
                        'input_image': img_str,
                    },
                ).json()

                print(request)
                request_id = request["id"]
                polling_url = request["polling_url"] # Use this URL for polling

            count_time = 0
            while True:
                time.sleep(0.5)
                result = requests.get(
                    polling_url,
                    headers={
                        'accept': 'application/json',
                        'x-key': os.environ.get("BFL_API_KEY"),
                    },
                    params={'id': request_id}
                ).json()
                
                if result['status'] == 'Ready':
                    print(f"Image ready: {result['result']['sample']}")
                    # A successful response will be a json object containing the result, and result['sample'] is a signed URL for retrieval
                    response = result['result']['sample']
                    # save the image, response is a signed URL for retrieval
                   
                    # Fetch the image content
                    img_data = requests.get(response).content
                    save_path = './flux/multipass/' + str(image_idx)+f'_input_raw_turn_{turns}.png'
                    # Save to file
                    with open(save_path, "wb") as f:
                        f.write(img_data)
                    break
                elif result['status'] in ['Error', 'Failed']:
                    print(f"Generation failed: {result}")
                    break
                count_time += 1
                if count_time > 100:
                    print(f"Generation failed: {result}")
                    break


    except Exception as e:
        print('Error:')
        print(e)
        print('idx', kk)
        continue






