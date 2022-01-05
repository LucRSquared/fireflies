# Fireflies

This notebook is inspired by the interactive page 'Fireflies' by Nicky Case (https://ncase.me/fireflies/) illustrating how a group of fireflies can spontaneously synchronize without a 'leader'.

This implementation is for a submission to Matt Parker's (@standupmaths) video *My 500-LED xmas tree got into Harvard* (https://www.youtube.com/watch?v=WuMRJf6B5Q4)

This notebook simulates the behavior of fireflies using the LEDs of Matt's tree and generates a `.csv` file that will transcribe the RGB values of each LED.

## Synchronization

The elementary principle is that each firefly in the swarm possesses its own internal 'clock' and it flashes when the clock strikes 'twelve'. If another firefly sees it, it slightly nudges its own clock forward. After a while, groups of fireflies can see their flash pulses synchronize

## Code

Please refer to the `nb_lucioles.ipynb` notebook for the details.

## Binder

Click on the "launch binder" badge to play with the notebook yourself directly in your browser!
