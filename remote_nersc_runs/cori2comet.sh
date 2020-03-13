#!/bin/bash

if [ -n "$STY" ]; then
    echo "Please press CTRL+A+D to detach from this screen session while leaving it open in background."
fi

start_time="$(date -u +%s)"

# --------------------------
# - file transfer settings
# --------------------------

logfile="rsync-log.txt" # will be attached to the email
delay=3 # minutes of delay between the attempts
source=tyson1@cori.nersc.gov:/global/cscratch1/sd/tyson1/projects/blending/buzzard_v2.0.0_lsst/zsnb/r3/zsnb.*.pickle
dest=/oasis/projects/nsf/ddp340/tyson1/buzzard_v2.0.0_lsst_r3/

# create the destination dir in sdsc if it does not exist
mkdir -p "$dest"

while [ 1 ]
do
    rsync -avz --partial "$source" "$dest" --log-file="$logfile" # -p will re-transfer the files from where it was left off
    if [ "$?" = "0" ] ; then
        echo "Rsync completed successfully!"
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

subject="[$$-$RANDOM] File transfer completed successfully"

body=''
body+="Hi Tony and Erfan,\n\n"
body+=$(printf "Congratulations! Your file transfer from NERSC to SDSC has completed successfully in %02d:%02d:%02d hms. " $(($secs/3600)) $(($secs%3600/60)) $(($secs%60)))
body+="Please have a look at the attached file $logfile."
body+="\n\nSent by Erfan's automated script on GitHub :)"

echo -e "$body" | mailx -s "$subject" -a "$logfile" "tyson@physics.ucdavis.edu erfan@ucdavis.edu"

# -------------------------------
# - usage [screen is optional]:
# -------------------------------
# screen -S myscreen (to start)
# bash <(wget --no-cache -qO- https://github.com/cosmicshear/BlendSim/raw/master/remote_nersc_runs/cori2comet.sh)
# CTRL + A + D (to detach)
# screen -X -S myscreen quit (to kill)

# -------------------------------
# - used ref:
# -------------------------------
# https://unix.stackexchange.com/questions/48298/can-rsync-resume-after-being-interrupted
# https://stackoverflow.com/questions/8260858/how-to-send-email-from-terminal

