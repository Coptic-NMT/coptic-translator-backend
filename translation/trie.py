

class TrieNode:

    def __init__(self, char, value = None, children=None):
        self.char = char
        self.children = children or {}
        self.value = value

    def __repr__(self):
        return f"TrieNode({self.char}, {self.value}, {self.children})"
    

class Trie:

    def __init__(self):
        self.root = TrieNode("")
        self.size = 0
        self.max_word_length = 0

    def add_word(self, word, value):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode(char)
            node = node.children[char]
        node.value = value
        self.size += 1
        self.max_word_length = max(self.max_word_length, len(word))

    def find_largest_substitution(self, text, start_index):
        node = self.root
        curr_word = ''
        last_word = ''
        last_substitution = ""
        for char in text[start_index:start_index + self.max_word_length]:
            if char not in node.children:
                return last_word, last_substitution
            node = node.children[char]
            curr_word += char
            if node.value is not None:
                last_word = curr_word
                last_substitution = node.value
        return last_word, last_substitution
    

if __name__ == "__main__":
    trie = Trie()
    trie.add_word("sub", "smaller")
    trie.add_word("substitute", "medium1")
    trie.add_word("substitution", "medium2")
    trie.add_word("substitutional", "biggest")
    text = "substitutional subs substitutiona"
    for i in range(len(text)):
        print(text[i:], trie.find_largest_substitution(text, i))
