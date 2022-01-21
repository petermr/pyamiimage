
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
        
