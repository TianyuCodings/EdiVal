import json
import os

def extract_unique_attributes(directory_path):
    """
    Extract unique objects, colors, and materials from all JSON files in the directory.
    """
    objects = set()
    colors = set()
    materials = set()
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    processed_files = 0
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if the file has parsed_json section
            if 'parsed_json' in data:
                parsed_json = data['parsed_json']
                
                # Iterate through each item in parsed_json
                for key, value in parsed_json.items():
                    # Skip special keys that don't contain object data
                    if key in ['All Objects', 'Filtered All Objects']:
                        continue
                    
                    # Extract attributes if they exist and are valid
                    if isinstance(value, dict):
                        if 'object' in value and value['object']:
                            objects.add(value['object'].strip())
                        
                        if 'color' in value and value['color']:
                            # Split colors by "and" and add each individually
                            color_parts = [c.strip() for c in value['color'].split(' and ')]
                            for color_part in color_parts:
                                if color_part:  # Only add non-empty strings
                                    colors.add(color_part)
                        
                        if 'material' in value and value['material']:
                            # Split materials by "and" and add each individually
                            material_parts = [m.strip() for m in value['material'].split(' and ')]
                            for material_part in material_parts:
                                if material_part:  # Only add non-empty strings
                                    materials.add(material_part)
            
            processed_files += 1
            if processed_files % 100 == 0:
                print(f"Processed {processed_files} files...")
                
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"Successfully processed {processed_files} files")
    print(f"Found {len(objects)} unique objects, {len(colors)} unique colors, {len(materials)} unique materials")
    
    return objects, colors, materials

def save_to_file(items, filename):
    """
    Save a set of items to a text file, one item per line, sorted alphabetically.
    """
    sorted_items = sorted(items)
    with open(filename, 'w', encoding='utf-8') as f:
        for item in sorted_items:
            f.write(item + '\n')
    print(f"Saved {len(sorted_items)} items to {filename}")

def main():
    # Directory containing the JSON files
    directory_path = '/scratch/EditVal/generate_instructions/grounding_all_objects'
    
    # Extract unique attributes
    objects, colors, materials = extract_unique_attributes(directory_path)
    
    # Save to text files
    save_to_file(objects, 'object_names.txt')
    save_to_file(colors, 'colors.txt')
    save_to_file(materials, 'materials.txt')
    
    print("\nExtraction complete!")
    print(f"- object_names.txt: {len(objects)} unique objects")
    print(f"- colors.txt: {len(colors)} unique colors")
    print(f"- materials.txt: {len(materials)} unique materials")

if __name__ == "__main__":
    main()
