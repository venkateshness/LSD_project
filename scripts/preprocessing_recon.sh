
# Run a) LSD Music
python3 preprocessing.py -s 01 02 03 04 05 06 07 08 09 10 11 -t Music -d LSD
wait

# Run b) PLA Music
python3 preprocessing.py -s 01 02 03 04 05 06 07 08 09 10 11 -t Music -d PLA
wait

# Run c) LSD Video
python3 preprocessing.py -s 01 02 03 04 05 06 07 08 09 10 11 -t Video -d LSD
wait

# Run d) PLA Video
python3 preprocessing.py -s 01 02 03 04 05 06 07 08 09 10 11 -t Video -d PLA
wait

echo "All preprocessing conditions completed!"
