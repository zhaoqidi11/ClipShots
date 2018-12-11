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


    def CTDetectionBaseOnHist(self, CandidateSegments, Truth):
        import numpy as np
        import cv2

        k = 0.4
        Tc = 0.05

        CandidateSegments = CandidateSegments

        HardCutTruth = Truth

        # It save the predicted shot boundaries
        Answer = []

        # It save the False Cut
        FalseCut = []
        
        # It save the Missed Cut1(Because the peak is too small)
        MissedCut1 = []
        # It save the Missed Cut2(Because the difference of first frame and the last frame in a segment is too small)
        MissedCut2 = []

        i_Video = cv2.VideoCapture(self.Video_path)

        # get width of this video
        wid = int(i_Video.get(3))

        # get height of this video
        hei = int(i_Video.get(4))
        AnswerLength = 0

        for i in range(len(CandidateSegments)):

            i_Video.set(1, CandidateSegments[i][0])
            ret1, frame1 = i_Video.read()

            i_Video.set(1, CandidateSegments[i][1])
            ret1, frame2 = i_Video.read()
            HistDifference = []


            if self.getHist(frame1, frame2, wid*hei)>0.5:
                for j in range(CandidateSegments[i][0], CandidateSegments[i][1]):

                    i_Video.set(1, j)
                    ret1_, frame1_ = i_Video.read()

                    i_Video.set(1, j+1)
                    ret2_, frame2_ = i_Video.read()

                    HistDifference.append(self.getHist_chi_square(frame1_, frame2_, wid*hei))


                if np.max(HistDifference) > 0.1 and len([_ for _ in HistDifference if _>0.1])<len(HistDifference):
                    CandidatePeak = -1
                    MAXValue = -1

                    if HistDifference[0] > 0.1 and HistDifference[0] > HistDifference[1]:
                        CandidatePeak = 0
                        MAXValue = HistDifference[0] - HistDifference[1]

                    for ii in range(1,len(HistDifference)-1):
                        if HistDifference[ii]>0.1 and HistDifference[ii] > HistDifference[ii-1] and HistDifference[ii] > HistDifference[ii+1]:
                            if np.max([np.abs(HistDifference[ii]-HistDifference[ii-1]), np.abs(HistDifference[ii]-HistDifference[ii+1])])>MAXValue:
                                CandidatePeak = ii
                                MAXValue = np.max([np.abs(HistDifference[ii]-HistDifference[ii-1]), np.abs(HistDifference[ii]-HistDifference[ii+1])])

                    if HistDifference[-1] > 0.1 and HistDifference[-1] > HistDifference[-2] and (HistDifference[-1]-HistDifference[-2])>MAXValue:
                        CandidatePeak = len(HistDifference)-1
                        MAXValue = HistDifference[-1]-HistDifference[-2]
                    if MAXValue>-1:
                        Answer.append(([CandidateSegments[i][0]+CandidatePeak, CandidateSegments[i][0]+CandidatePeak+1]))


                if len(Answer) > 0 and len(Answer) > AnswerLength:
                    AnswerLength += 1
                    if Answer[-1] not in HardCutTruth:
                        FalseCut.append[Answer[-1]]
                    # Flag = False
                    # for k in HardCutTruth:
                    #     Flag = self.if_overlap(Answer[-1][0], Answer[-1][1], k[0], k[1])
                    #     if Flag:
                    #         break
                    # if Flag is False:
                    #     print 'This is a false cut: ', Answer[-1]
                else:
                    for k1 in HardCutTruth:
                        if self.if_overlap(CandidateSegments[i][0], CandidateSegments[i][1], k1[0], k1[1]):
                            MissedCut1.append[k1]

            else:
                for k2 in HardCutTruth:
                    if self.if_overlap(CandidateSegments[i][0], CandidateSegments[i][1], k2[0], k2[1]):
                        MissedCut2.append[k2]
        Miss = 0
        True_ = 0
        False_ = 0
        for i in Answer:
            if i not in HardCutTruth:
                print 'False :', i, '\n'
                False_ = False_ + 1
            else:
                True_ = True_ + 1

        for i in HardCutTruth:
            if i not in Answer:
                Miss = Miss + 1

        print 'False No. is', False_,'\n'
        print 'True No. is', True_, '\n'
        print 'Miss No. is', Miss, '\n'


# Change the annotation of ClipShots to the annotation of Dataset
if __name__ == '__main__':
    import math

    os.chdir('D:\\ClipShots\\ClipShots\\ClipShots\\annotations')
    annotations = json.load(open('./train.json'))
    os.chdir('D:\\ClipShots\\ClipShots\\ClipShots\\converted_annotations_test')
    VideoNames = []
    Labels = []

    test1 = ShotDetection()

    for videonames, labels in annotations.items():
        #labelsnew = ['\t'.join([str(i[0]), str(i[1])+'\n']) for i in labels['transitions']]
        #with open('.'.join([str(videonames).split('.')[0]+'_gt','txt']), 'w') as f:
        #          f.writelines(labelsnew)
        Labels = [i for i in labels['transitions']]
        
        HardLabels = []
        GraLabels = []

        test1.SetVideo_path(str(videonames))
        #CandidateSegments = test1.CutVideoIntoSegments()
        
        CandidateHardLabels=[]

        for i in Labels:
            if i[1]-i[0] == 1:
                HardLabels.append(i)
            else:
                GraLabels.append(i)
    
        CandidateHardLabels = [[math.floor(cut[0]/10.0), math.ceil(cut[1]/10.0)] for cut in HardLabels]
        CandidateSegments = test1.CutVideoIntoSegments()
        NewCandidateSegments = []
        for i in CandidateHardLabels:
            if i not in CandidateSegments:



        HardLabelsLength = float(len(HardLabels))
        GraLabelsLength = float(len(GraLabels))

        AllHardLabels += HardLabelsLength
        AllGraLabels += GraLabelsLength



        test1.CTDetectionBaseOnHist(CandidateHardLabels, HardLabels)


