import csv
import re

# this code was run only one time
# to make every song as a line
# making each word lowercase and without punctuations
# and storing it in formattedSongs.txt


def get_word(word):
    return re.sub(r"[^a-zA-Z0-9]+", '', word.lower())


def main():
    with open('data/songdata.csv', 'r') as csv_file:
        with open("data/formattedSongs.txt", 'a') as target:
            reader = csv.reader(csv_file)
            next(reader)
            for header in reader:
                s = header[3]
                words = list(map(get_word, s.split()))
                for word in words:
                    target.write(word + " ")
                target.write("\n")


if __name__ == '__main__':
    main()
