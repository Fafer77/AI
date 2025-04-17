# idea - programowanie dynamiczne, mamy koszt - maksymalizacja sumy kwadratów długości słów

def is_word(candidate: str, words: set[str]) -> bool:
    return candidate in words


def read_set_words(filename: str = 'lista1/polish_words.txt') -> set[str]:
    words = set()
    with open(filename, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            words.add(line.strip())
    
    return words


def seperate_text(words_set: set[str], filename: str = 'zad2_input.txt'):
    texts = []
    with open(filename, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            texts.append(line.strip())
    
    seperated_texts = []

    for text in texts:
        words = []
        n = len(text)
        dp = [-1] * (n + 1)
        indices = [-1] * (n + 1)

        dp[0] = 0

        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] != -1 and is_word(text[j:i], words_set):
                    word_len = i - j
                    candidate = dp[j] + word_len * word_len
                    if candidate > dp[i]:
                        dp[i] = candidate
                        indices[i] = j
        
        k = n
        while k > 0:
            l = indices[k]
            words.append(text[l:k])
            k = l
        words.reverse()
        seperated_texts.append(words)
    
    return seperated_texts
        

if __name__ == '__main__':
    words_set = read_set_words()
    result = seperate_text(words_set, 'lista1/pan_tadeusz_bez_spacji.txt')
    file = open('pan_tadeusz_max_squares.txt', 'w', encoding='utf-8')
    for words in result:
        file.write(" ".join(words))
        file.write('\n')
    
