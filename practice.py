def merge_alternately(word1, word2):
    reverse = []
    
    i = 1
    
    while i <= len(word1):
        reverse.append(word1[len(word1) - i])
        i += 1
    return "".join(reverse)
    


if __name__ == "__main__":
    word1 = input().strip()
    word2 = input().strip()
    print(merge_alternately(word1, word2))
