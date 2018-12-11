import os
import json

class ShotDetection():

    def SetVideo_path(self, Video_path):
        self.Video_path = 'D:\\ClipShots\\ClipShots\\ClipShots\\videos\\train\\'+Video_path

    def CutVideoIntoSegments(self):
        import math
        import cv2
        import numpy as np

        # It save the pixel intensity between 20n and 20(n+1)
        d = []
        SegmentsLength = 11
        i_Video = cv2.VideoCapture(self.Video_path)

        # get width of this video
        wid = int(i_Video.get(3))

        # get height of this video
        hei = int(i_Video.get(4))

        if i_Video.isOpened():
            success = True
        else:
            success = False
            print('Can\' open this video!')

        # It save the number of frames in this video
        FrameNumber = int(i_Video.get(7))

        # The number of segments
        Count = int(math.ceil(float(FrameNumber) / float(SegmentsLength)))
        for i in range(Count):

            i_Video.set(1, (SegmentsLength-1)*i)
            ret1, frame_20i = i_Video.read()

            if((SegmentsLength-1)*(i+1)) >= FrameNumber:
                i_Video.set(1, FrameNumber-1)
                ret2, frame_20i1 = i_Video.read()
                # d.append(np.sum(np.abs(self.RGBToGray(frame_20i) - self.RGBToGray(frame_20i1))))

                d.append(self.getHist(frame_20i, frame_20i1, wid*hei))
                break

            i_Video.set(1, (SegmentsLength-1)*(i+1))
            ret2, frame_20i1 = i_Video.read()

            # d.append(np.sum(np.abs(self.RGBToGray(frame_20i) - self.RGBToGray(frame_20i1))))
            d.append(self.getHist(frame_20i, frame_20i1, wid*hei))



        # The number of group
        GroupNumber = int(math.ceil(float(FrameNumber) / 10.0))

        MIUG = np.mean(d)
        a = 0.5 # The range of a is 0.5~0.7
        Tl = [] # It save the Tl of each group
        CandidateSegment = []
        for i in range(GroupNumber):



            MIUL = np.mean(d[10*i:10*i+10])
            SigmaL = np.std(d[10*i:10*i+10])

            Tl.append(MIUL + a*(1+math.log(MIUG/MIUL))*SigmaL)
            for j in range(10):
                if i*10 + j >= len(d):
                    break
                if d[i*10+j]>Tl[i]:
                    CandidateSegment.append([(i*10+j)*(SegmentsLength-1), (i*10+j+1)*(SegmentsLength-1)])
                    #print 'A candidate segment is', (i*10+j)*20, '~', (i*10+j+1)*20


        for i in range(1,len(d)-1):
            if (d[i]>(3*d[i-1]) or d[i]>(3*d[i+1])) and d[i]> 0.8 * MIUG:
                if [i*(SegmentsLength-1), (i+1)*(SegmentsLength-1)] not in CandidateSegment:
                    j = 0
                    while j < len(CandidateSegment):
                        if (i+1)*(SegmentsLength-1)<= CandidateSegment[j][0]:
                            CandidateSegment.insert(j, [i*(SegmentsLength-1), (i+1)*(SegmentsLength-1)])
                            break
                        j += 1
        return CandidateSegment



# Change the annotation of ClipShots to the annotation of Dataset
if __name__ == '__main__':
    os.chdir('D:\\ClipShots\\ClipShots\\ClipShots\\annotations')
    annotations = json.load(open('./test.json'))
    os.chdir('D:\\ClipShots\\ClipShots\\ClipShots\\converted_annotations_test')
    VideoNames = []
    Labels = []

    test1 = ShotDetection()


    for videonames, labels in annotations.items():
        #labelsnew = ['\t'.join([str(i[0]), str(i[1])+'\n']) for i in labels['transitions']]
        #with open('.'.join([str(videonames).split('.')[0]+'_gt','txt']), 'w') as f:
        #          f.writelines(labelsnew)
        Labels = [i for i in labels['transitions']]

        test1.SetVideo_path(str(videonames))
