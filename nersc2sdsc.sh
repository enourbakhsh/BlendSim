#!/bin/bash

start_time="$(date -u +%s)"

# --------------------------
# - file transfer settings
# --------------------------

logfile="screen.txt" # will be attached to the email
delay=3 # minutes of delay between the attempts
source=tyson1@cori.nersc.gov:/global/cscratch1/sd/tyson1/projects/blending/buzzard_v1.9.2_lsst/zsnb/zsnb.*.fit
dest=/oasis/projects/nsf/ddp340/tyson1/buzzard_v1.9.2_lsst_r3/

while [ 1 ]
do
    rsync -avz --partial "$source" "$dest" --log-file="$logfile" # -p will re-transfer the files from where it was left off
    if [ "$?" = "0" ] ; then
        echo "rsync completed successfully!"
        break
    else
        echo "Rsync failure. Backing off and retrying in $delay minutes ..."
        sleep "$delay"m
    fi
done

end_time="$(date -u +%s)"
secs="$(($end_time-$start_time))"

# ------------------
# - email settings
# ------------------

subject="[pid=$$-$RANDOM] File transfer completed successfully"

body=''
body+="Hi Tony and Erfan,\n\n"
body+=$(printf "Congratulations! Your file transfer from NERSC to SDSC has completed successfully in %02d:%02d:%02d hms." $(($secs/3600)) $(($secs%3600/60)) $(($secs%60)))
body+="\n\nSent by Erfan's automated script on GitHub :)"

echo -e "$body" | mailx -s "$subject" -a "$logfile" "tyson@physics.ucdavis.edu erfan@ucdavis.edu"

# - usage:
# bash <(wget --no-cache -qO- https://github.com/cosmicshear/BlendSim/raw/master/nersc2sdsc.sh)

# - used ref:
# https://unix.stackexchange.com/questions/48298/can-rsync-resume-after-being-interrupted
# https://stackoverflow.com/questions/8260858/how-to-send-email-from-terminal

