# +
import os
import argparse
import inflect
try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    use_ttsfrd = False
from .frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph

class CosyVoiceNormalizer:
    def __init__(self):
        self.use_ttsfrd = use_ttsfrd
        self.inflect_parser = inflect.engine()
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize('{}/../../pretrained_models/CosyVoice-ttsfrd/resource'.format(ROOT_DIR)) is True, 'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyin')
            self.frd.enable_pinyin_mix(True)
            self.frd.set_breakmodel_index(1)
        else:
            self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False)
            self.en_tn_model = EnNormalizer()
    
    def text_normalize(self, text):
        text = text.strip()
        if contains_chinese(text):
            if self.use_ttsfrd:
                text = self.frd.get_frd_extra_info(text, 'input')
            else:
                text = self.zh_tn_model.normalize(text)
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "、")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
        else:
            if self.use_ttsfrd:
                text = self.frd.get_frd_extra_info(text, 'input')
            else:
                text = self.en_tn_model.normalize(text)
            text = spell_out_number(text, self.inflect_parser)

        return text

def main(args):
    normalizer = CosyVoiceNormalizer()
    if args.text:
        print(normalizer.text_normalize(args.text))
    if args.file:
        assert args.output_path
        with open(args.file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

            # Process each line
            normalized_lines = []
            for line in lines:
                if not line.strip():
                    continue  # Skip empty lines

                # Split the line into filename and text content
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    print(f"Skipping malformed line: {line}")
                    continue

                filename, text = parts
                # Normalize the text content
                normalized_text = normalizer.text_normalize(text)
                # Reconstruct the line with the normalized text
                normalized_lines.append(f"{filename} {normalized_text}")
        # Write the normalized lines to the output file
        with open(args.output_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(normalized_lines))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', type=str)
    parser.add_argument('-f', '--file', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    args = parser.parse_args()
    main(args)
