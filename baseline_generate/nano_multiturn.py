import pandas as pd
import ast
import os
import os
import io
import PIL.Image
import PIL
from tqdm import tqdm
import base64
import mimetypes
import os
from google import genai
from google.genai import types

root_dir = './input_images_resize_512/'
csv_path = './oai_instruction_generation_output.csv'
df = pd.read_csv(csv_path)
df = df[df['turns'] == 3]
print(df.iloc[0])



os.environ["GEMINI_API_KEY"] = '<YOUR_API_KEY_HERE>'


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to to: {file_name}")


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-image-preview"
    for kk in range(len(df)):
        try:
            chat = client.chats.create(
            model=model,
            )

            raw_record = df.iloc[kk]
            prompt = (raw_record['instructions'])
            prompt = ast.literal_eval(prompt)
            prompt = [promp.strip('.').strip('"').strip('\'') for promp in prompt]
            image_idx = raw_record['image_index']
            image_path = os.path.join(root_dir, str(image_idx)+'_input_raw.jpg')
            image = PIL.Image.open(image_path)

            for turn in [1,2,3]:
                turns = str(turn)
                prompt_i = prompt[turn-1]
                print(prompt_i)
                if turn == 1:
                    response = chat.send_message([image, prompt_i])
                else:
                    response = chat.send_message(prompt_i)
                file_index = 0
                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        print(part.text)
                    elif part.inline_data is not None:
                        file_index += 1
                        inline_data = part.inline_data
                        image_output = PIL.Image.open(io.BytesIO(inline_data.data))
                        image_output = image_output.resize((512, 512))
                        save_path = './nano_banana/multipass/' + str(image_idx)+f'_input_raw_turn_{turns}.png'
                        image_output.save(save_path)
                        print('save_path', save_path)
        except Exception as e:
            print('Error:')
            print(e)
            print('idx', kk)
            continue
            


if __name__ == "__main__":
    generate()















