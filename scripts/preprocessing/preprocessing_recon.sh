
# Run a) LSD Music
python3 preprocessing.py -s 005 016 010 013 015 009 018 003 017 006 011 -t RestO -d LSD
wait

# Run b) PLA Music
python3 preprocessing.py -s 005 016 010 013 015 009 018 003 017 006 011 -t RestO -d PLA
wait

# Run c) LSD Video
python3 preprocessing.py -s 005 016 010 013 015 009 018 003 017 006 011 -t RestC -d LSD
wait

# Run d) PLA Video
python3 preprocessing.py -s 005 016 010 013 015 009 018 003 017 006 011 -t RestC -d PLA
wait

echo "All preprocessing conditions completed!"
