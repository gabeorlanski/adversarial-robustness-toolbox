# Sampling Youtube ID's for a set of labels kinetics 400 csv files
# Download Youtube videos with links from a csv file

# Run this code in the folder where you want to store the videos.

import pandas as pd
from pytube import YouTube

if __name__ == "__main__":
    
    # Read csv file
    links = pd.read_csv("sampled_urls.csv")
    # Download the videos via pytube library
    for i in range(0, len(links.youtube_id)):
        name = links.youtube_id[i]
        print(name)
        link = "https://youtu.be/" + str(name)
        print(link)
        yt = YouTube(link)
        video = yt.streams
        print(video)
        video.first().download(filename = name)
    print("Finished")
    