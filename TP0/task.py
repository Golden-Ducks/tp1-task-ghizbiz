#zerhouni ghizlene houria
n= {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"
}
def normalisation(texte):

    texte = texte.lower()
    mots= texte.split()
    mots_normalisés= []
    for w in mots:
        if w.isdigit() and w in n:
            mots_normalisés.append(n[w])
        else:
            mots_normalisés.append(w)
    return mots_normalisés
D1 = "Today she cooked 4 bourak. Later, she added two chamiyya and 1 pizza."
D2 = "Five pizza were ready, but 3 bourak burned!"
D3 = "We only had 8 chamiyya, no pizza, and one tea."
D4 = "Is 6 too much? I ate nine bourak lol."
docs = [D1, D2, D3, D4]
for i, d in enumerate(docs, 1):
    print(f"D{i} normalized: {normalisation(d)}")
