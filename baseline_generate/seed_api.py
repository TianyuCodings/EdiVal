# Please go to https://fal.ai/ to your API key and set it in your environment variables as 
# 

import fal_client

def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
           print(log["message"])




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

os.environ['FAL_API_KEY'] = 'your_api_key_here'
root_dir = '../image_generate/input_images_resize_512/'
csv_path = '../image_generate/oai_instruction_generation_output.csv'
df = pd.read_csv(csv_path)
df = df[df['turns'] == 3]
print(df.iloc[0])



# for kk in range(len(df)):
for kk in range(len(df)):
    try:
        raw_record = df.iloc[kk]
        prompt = (raw_record['instructions'])
        prompt = ast.literal_eval(prompt)
        prompt = [promp.strip('.').strip('"').strip('\'') for promp in prompt]
        image_idx = raw_record['image_index']
        image_path = os.path.join(root_dir, str(image_idx)+'_input_raw.jpg')
        image_path = fal_client.upload_file(image_path)
         

        for turn in [1,2,3]:
            turns = str(turn)
            prompt_i = prompt[turn-1]
            print(prompt_i)
            if turn == 1:
                result = fal_client.subscribe(
                    "fal-ai/bytedance/seedream/v4/edit",
                    arguments={
                        "prompt": prompt_i,
                        "image_urls": [image_path]
                    },
                    with_logs=True,
                    on_queue_update=on_queue_update,
                )
                print(result)
            else:
                previous_image_path = '/home/ubuntu/MIT-UCB/yasi-project/edival_user/api/seedream/multipass/' + str(image_idx)+f'_input_raw_turn_{turn-1}.png'
                previous_image_path = fal_client.upload_file(previous_image_path)
                result = fal_client.subscribe(
                    "fal-ai/bytedance/seedream/v4/edit",
                    arguments={
                        "prompt": prompt_i,
                        "image_urls": [previous_image_path]
                    },
                    with_logs=True,
                    on_queue_update=on_queue_update,
                )
                print(result)
            url = result['images'][0]['url']
            img_data = requests.get(url).content
            save_path = '/home/ubuntu/MIT-UCB/yasi-project/edival_user/api/seedream/multipass/' + str(image_idx)+f'_input_raw_turn_{turns}.png'
            # Save to file
            with open(save_path, "wb") as f:
                f.write(img_data)

    except Exception as e:
        print('Error:')
        print(e)
        print('idx', kk)
        continue