import os
import json

class ShotDetection():
    # Set the Video path(it is only includes the name of video)
    def SetVideo_path(self, Video_path):
        self.Video_path = 'D:\\ClipShots\\ClipShots\\ClipShots\\videos\\train\\'+Video_path



    # Get the Manhattan Distance
    def Manhattan(self, vector1, vector2):
        import numpy as np
        return np.sum(np.abs(vector1 - vector2))

    def getHist(self, frame1, frame2, allpixels):
        binsnumber = 64
        import cv2
        Bframe1hist = cv2.calcHist([frame1], channels=[0], mask=None, ranges=[0.0,255.0], histSize=[binsnumber])
        Bframe2hist = cv2.calcHist([frame2], channels=[0], mask=None, ranges=[0.0,255.0], histSize=[binsnumber])

        Gframe1hist = cv2.calcHist([frame1], channels=[1], mask=None, ranges=[0.0,255.0], histSize=[binsnumber])
        Gframe2hist = cv2.calcHist([frame2], channels=[1], mask=None, ranges=[0.0,255.0], histSize=[binsnumber])

        Rframe1hist = cv2.calcHist([frame1], channels=[2], mask=None, ranges=[0.0,255.0], histSize=[binsnumber])
        Rframe2hist = cv2.calcHist([frame2], channels=[2], mask=None, ranges=[0.0,255.0], histSize=[binsnumber])

        distance = self.Manhattan(Bframe1hist, Bframe2hist) + self.Manhattan(Gframe1hist, Gframe2hist) + self.Manhattan(Rframe1hist, Rframe2hist)
        return distance/(allpixels)        

    # Check the transition[begin1,end1] and transition[begin2, end2] whether has overlap
    def if_overlap(self, begin1, end1, begin2, end2):
        if begin1 > begin2:
            begin1, end1, begin2, end2 = begin2, end2, begin1, end1

        return end1 >= begin2

    def CheckCandidateSegments(self, CutTruth, CandidateSegments):
        Miss = []
        for i in range(len(CutTruth)):
            for j in range(len(CandidateSegments)):
                if self.if_overlap(CutTruth[i][0], CutTruth[i][1], CandidateSegments[j][0], CandidateSegments[j][1]):
                    break
                if j == len(CandidateSegments) - 1:
                    Miss.append(CutTruth[i])

        return Miss




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
        a = 0.3 # The range of a is 0.5~0.7
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
    annotations = json.load(open('./train.json'))
    os.chdir('D:\\ClipShots\\ClipShots\\ClipShots\\converted_annotations_test')
    VideoNames = []
    Labels = []

    test1 = ShotDetection()

    AllHardLabels = 0
    AllGraLabels = 0
    AllMissedHardLabels = 0
    AllMissedGraLabels = 0
    for videonames, labels in annotations.items():
        #labelsnew = ['\t'.join([str(i[0]), str(i[1])+'\n']) for i in labels['transitions']]
        #with open('.'.join([str(videonames).split('.')[0]+'_gt','txt']), 'w') as f:
        #          f.writelines(labelsnew)
        Labels = [i for i in labels['transitions']]
        
        HardLabels = []
        GraLabels = []

        test1.SetVideo_path(str(videonames))
        CandidateSegments = test1.CutVideoIntoSegments()
        
        for i in Labels:
            if i[1]-i[0] == 1:
                HardLabels.append(i)
            else:
                GraLabels.append(i)
    
        HardLabelsLength = float(len(HardLabels))
        GraLabelsLength = float(len(GraLabels))

        AllHardLabels += HardLabelsLength
        AllGraLabels += GraLabelsLength

        MissedHardLabels = len(test1.CheckCandidateSegments(HardLabels, CandidateSegments))
        MissedGraLabels = len(test1.CheckCandidateSegments(GraLabels, CandidateSegments))

        AllMissedHardLabels += MissedHardLabels
        AllMissedGraLabels += MissedGraLabels
        
        if HardLabelsLength>0:
            print ' The rate of This video Candidate segments including true hard cut is', (HardLabelsLength-MissedHardLabels)/HardLabelsLength
        if GraLabelsLength>0:
            print ' The rate of This video Candidate segments including true Gra cut is', (GraLabelsLength-MissedGraLabels)/GraLabelsLength
        
    print 'All True Hard Cut is ', AllHardLabels
    print 'All True Gra is', AllGraLabels
    print 'All Missed Hard Cut is ', AllMissedHardLabels
    print 'All Missed Gra Cut is ', AllMissedGraLabels


