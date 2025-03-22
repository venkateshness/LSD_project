
# Run a) LSD Music
python3 source_localization.py -s 003 005 006 009 010 013 015 016 017 018 -t Music -d LSD  > LSD_Music.log 2>&1
wait

# Run b) PLA Music
python3 source_localization.py -s 003 005 006 009 010 013 015 016 017 018 -t Music -d PLA  > PLA_Music.log 2>&1
wait

# #video is not run - Jan 7
# # Run c) LSD Video
# python3 source_localization.py -s 005 016 010 013 015 009 018 003 017 006 -t Video -d LSD
# wait

# # Run d) PLA Video
# python3 source_localization.py -s 005 016 010 013 015 009 018 003 017 006  -t Video -d PLA
# wait

