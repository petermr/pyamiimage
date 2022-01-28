
from audioop import reverse


class WordCleaner:
    
    special_char = ['|', '>', '<', '-', '_', ',', '.', '`',
                     ';', ':', '\'', '=', '€', '~', '*', '!',
                     '$', '%', '^', '&', '(', ')', '/', '\\']    
    frequently_misread_letters = ['Y']

    @classmethod
    def remove_single_special_characters(cls, textboxes):
        """
        Remove all special characters from a list
        :param: textboxes is a list of TextBox objects
        :returns: list of TextBox objects 
        """
        return [item for item in textboxes if item.get_text() not in cls.special_char] 

    @classmethod
    def remove_all_single_characters(cls, textboxes):
        """
        Removes all single characters in a list of textboxes
        """
        return [item for item in textboxes if len(item.get_text()) > 1 ]

    @classmethod
    def remove_all_sequences_of_special_characters(cls, textboxes):
        """
        Removes sequences of special characters of any length
        """
        textboxes_cleaned = []
        for textbox in textboxes:
            special_character = True
            for letter in textbox.get_text():
                if letter.isalnum():
                    special_character = False
                    break
            if not special_character:
                textboxes_cleaned.append(textbox)
            
        return textboxes_cleaned

    @classmethod
    def remove_trailing_special_characters(cls, textboxes):
        for textbox in textboxes:
            appexdix_length = 0
            text = textbox.get_text()
            for c in reversed(text):
                if not c.isalnum():
                    text = text[:-1]
                    appexdix_length += 1
            textbox.set_text(text)
        return textboxes
        
    @classmethod
    def remove_leading_special_characters(cls, textboxes, bboxes):
        cleaned_textboxes = []
        cleaned_bboxes = []
        for textbox, bbox in zip(textboxes, bboxes):
            appexdix_length = 0
            text = textbox
            for c in text:
                if not c.isalnum():
                    textbox = textbox[1:]
                    appexdix_length += 1
            cleaned_textboxes.append(textbox)
            cleaned_bboxes.append(bbox)
        return cleaned_textboxes, cleaned_bboxes

    @classmethod
    def remove_numbers_only(cls, textboxes, bboxes):
        """remove all words which are composed of numbers and special characters"""
        cleaned_textbox = []
        cleaned_bboxes = []
        for textbox, bbox in zip(textboxes, bboxes):
            word = False
            for c in textbox:
                if c.isalpha():
                    word = True
                    break
            if word:
                cleaned_textbox.append(textbox)
                cleaned_bboxes.append(bbox)
        return cleaned_textbox, cleaned_bboxes

    @classmethod
    def remove_misread_letters(cls, textboxes, bboxes):
        """Remove whole words consisting only of frequently misread letters such as vV"""    
        cleaned_textbox = []
        cleaned_bboxes = []
        for textbox, bbox in zip(textboxes, bboxes):
            flag = False
            for c in textbox:
                if c not in cls.frequently_misread_letters:
                    flag = True
                    break
            if flag:
                cleaned_textbox.append(textbox)
                cleaned_bboxes.append(bbox)
        return cleaned_textbox, cleaned_bboxes
