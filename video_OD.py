import subprocess


def video2frame(input_dir,filename, fps_r):
    strcmd1 = "mkdir frames"
    strcmd2 = "mkdir frames_trans"
    strcmd3 = "ffmpeg -i " + input_dir + filename + " -vf fps=" + str(fps_r) + " frames/" + "frame-%04d.jpg"
    for i in [strcmd1,strcmd2, strcmd3]:
        print(i)
        subprocess.call(i, shell=True)

def frame2video(frame_dir, fps_r):
    strcmd = "ffmpeg -framerate " + str(fps_r) + " -i " + str(frame_dir) + "/frame-%04d.jpg -vcodec libx264 -b 800k video.avi"
    subprocess.call(strcmd, shell=True)

input_dir="demo/"
filename="Mission.avi"
video2frame(input_dir, filename,fps_r=20)

subprocess.call("python demo_for_video.py", shell=True)

#frame2video("frames_trans", 20)


def play_save_cap(input_dir, filename, fps=0):
    cap = cv2.VideoCapture(input_dir + filename)
    if fps==0:
        fps=int(cap.get(5))
    cap.set(cv2.CAP_PROP_FPS, fps)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('456.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 0)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # write the flipped frame
            print(out.isOpened())
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


    # Release everything if job is finished
    cap.release()
    out.release()