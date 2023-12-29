# computer_vision_hand
Leveraging opencv and mediapipe, there are a few projects here; the most interesting is FingerCountingProject.py which tracks hand landmarks and then
reproduces your finger postures, open or closed.  I experimented with algorithms to determine the curled state of each finger, most of them having to do
with calculating joint angles.  Ultimately I finally realized that I could just check the distance from a fingertop to the 0 landmark at the base of the
palm, and as soon as that distance is less than the distance from the finger base to the 0 landmark, let's call it curled.  The thumb is a little special,
but then I guess it's a pretty special digit--thanks, evolution.  
I also got to play with image compositing using opencv; the overlays I am doing are per finger, plus the palm, lower arm, and backdrop are all separate
elements.  I wound up using 24 bit pngs and a very helpful compositing method I found on stack overflow.  
I finally used the current time run through math.sin() to get a little bit of bob in the arm for spookiness' sake.  
