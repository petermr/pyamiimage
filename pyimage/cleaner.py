
class WordCleaner:
    
    special_char = ['|', '>', '<', '-', '_', ',', '.', '`',
                     ';', ':', '\'', '=', '€', '~', '*', '!',
                     '$', '%', '^', '&', '(', ')', '/', '\\']    
    frequently_misread_letters = ['Y', 'v', 'V']

    @classmethod
    def remove_single_special_characters(cls, textboxes):
        """
        Remove all special characters from a list
        :param: textboxes is a list of TextBox objects
        :returns: list of TextBox objects 
        """
        return [item for item in textboxes if item.text not in cls.special_char] 

    @classmethod
    def remove_all_single_characters(cls, textboxes):
        """
        Removes all single characters in a list of textboxes
        """
        return [item for item in textboxes if len(item.text) > 1 ]

    @classmethod
    def remove_all_sequences_of_special_characters(cls, textboxes):
        """
        Removes sequences of special characters of any length
        """
        textboxes_cleaned = []
        for textbox in textboxes:
            special_character = True
            for letter in textbox.text:
                if letter.isalnum():
                    special_character = False
                    break
            if not special_character:
                textboxes_cleaned.append(textbox)
            
        return textboxes_cleaned

    @classmethod
    def remove_trailing_special_characters(cls, textboxes):
        """removes all the special characters at the end of the word"""
        for textbox in textboxes:
            appexdix_length = 0
            text = textbox.text
            for c in reversed(text):
                if not c.isalnum():
                    text = text[:-1]
                    appexdix_length += 1
            textbox.set_text(text)
        return textboxes
        
    @classmethod
    def remove_leading_special_characters(cls, textboxes):
        """removes all the special characters in front of a word"""
        cleaned_textboxes = []
        for textbox in textboxes:
            appexdix_length = 0
            text = textbox.text
            for c in text:
                if not c.isalnum():
                    text = text[1:]
                    appexdix_length += 1
            cleaned_textboxes.append(textbox)
        return cleaned_textboxes

    @classmethod
    def remove_numbers_only(cls, textboxes):
        """remove all words which are composed of numbers and special characters"""
        cleaned_textbox = []
        for textbox in textboxes:
            word = False
            text = textbox.text
            for c in text:
                if c.isalpha():
                    word = True
                    break
            if word:
                cleaned_textbox.append(textbox)
        return cleaned_textbox

    @classmethod
    def remove_misread_letters(cls, textboxes):
        """Remove whole words consisting only of frequently misread letters such as vV"""    
        cleaned_textbox = []
        for textbox in textboxes:
            flag = False
            text = textbox.text
            for c in text:
                if c not in cls.frequently_misread_letters:
                    flag = True
                    break
            if flag:
                cleaned_textbox.append(textbox)
        return cleaned_textbox

    @classmethod
    def remove_leading_and_trailing_special_characters(cls, textboxes):
        """Convenience method for trimming special characters from words"""
        removed_leading = cls.remove_leading_special_characters(textboxes)
        removed_both = cls.remove_trailing_special_characters(removed_leading)
        return removed_both
