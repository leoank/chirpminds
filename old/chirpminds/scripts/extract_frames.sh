#!/bin/bash

sample_frames() {
    echo "Launching ffmpeg with $1 and out at $2"
    command ffmpeg -i "$1" \
                -ss 00:00:05 \
                -vframes 500 \
                -r 1 \
                "$2/frame%03d.jpg"
}


case "$1" in
  -s|--sample)
    sample_frames $2 $3
    ;;
  *)
    echo "Usage: (-s|--sample) [input file path] [output folder]"
    ;;
esac
