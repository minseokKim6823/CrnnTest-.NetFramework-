using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace CrnnTest
{
    public class DbNetDetector
    {
        private InferenceSession session;
        private int boxSizeThresh = 3;

        public DbNetDetector(string modelPath)
        {
            session = new InferenceSession(modelPath);
        }

 
        public List<Rectangle> Detect(Mat image)
        {
            int origW = image.Width;
            int origH = image.Height;
            int inputSize = 480;

            // CLAHE 적용 대비 향상
            Mat claheImg = new Mat();
            CvInvoke.CvtColor(image, claheImg, ColorConversion.Bgr2Gray);
            CvInvoke.CLAHE(claheImg, 2.0, new Size(8, 8), claheImg);

            // 2. Letterbox (비율 유지) 리사이징 + 패딩(검정색으로)
            float scale = Math.Min((float)inputSize / origW, (float)inputSize / origH);
            int resizedW = (int)(origW * scale);
            int resizedH = (int)(origH * scale);

            Mat resized = new Mat();
            CvInvoke.Resize(claheImg, resized, new Size(resizedW, resizedH));
            resized.ConvertTo(resized, DepthType.Cv32F, 1.0 / 255);
            // resized 된 이미지가 float32로 정규화되어 있음
            Image<Bgr, float> resizedImg = resized.ToImage<Bgr, float>();

            // 검정색 배경의 padded 이미지
            Image<Bgr, float> paddedImg = new Image<Bgr, float>(inputSize, inputSize);
            paddedImg.SetZero();

            // 중심 정렬 위치
            Rectangle roi = new Rectangle((inputSize - resizedW) / 2, (inputSize - resizedH) / 2, resizedW, resizedH);
            paddedImg.ROI = roi; 
            resizedImg.CopyTo(paddedImg); 
            paddedImg.ROI = Rectangle.Empty;

            // 추론용 데이터 준비 (NCHW: [1,3,960,960])
            var data = paddedImg.Data;
            float[] inputData = new float[3 * inputSize * inputSize];
            int idx = 0;
            for (int c = 0; c < 3; c++)
                for (int i = 0; i < inputSize; i++)
                    for (int j = 0; j < inputSize; j++)
                        inputData[idx++] = data[i, j, c];

            var tensor = new DenseTensor<float>(inputData, new[] { 1, 3, 480, 480 });

            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("x", tensor) };//학습 모델(ocr_ctc_model.onnx)은 input num모델은 x
            var results = session.Run(inputs);
            var output = results.First().AsTensor<float>();

            int rows = output.Dimensions[2];
            int cols = output.Dimensions[3];
            float[] flat = output.ToArray();

            // 4. Mask 생성
            Image<Gray, float> maskImage = new Image<Gray, float>(cols, rows);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    maskImage.Data[i, j, 0] = flat[i * cols + j];

            Mat mask = maskImage.Mat;

            // 5. Binary Threshold
            Mat binMask = new Mat();
            CvInvoke.Threshold(mask, binMask, 0.3, 255, ThresholdType.Binary);
            binMask.ConvertTo(binMask, DepthType.Cv8U);

            // 6. Contour Detection
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(binMask, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

            List<Rectangle> boxes = new List<Rectangle>();
            
           for (int i = 0; i < contours.Size; i++){
                Rectangle rect = CvInvoke.BoundingRectangle(contours[i]);
                if (rect.Width > boxSizeThresh && rect.Height > boxSizeThresh)
                {
                    int padding = 10;
                    Rectangle corrected = new Rectangle(
                        Math.Max(0, (int)((rect.X - roi.X) / scale) - padding),
                        Math.Max(0, (int)((rect.Y - roi.Y) / scale) - padding),
                        Math.Min(origW - (int)((rect.X - roi.X) / scale), (int)(rect.Width / scale) + 2 * padding),
                        Math.Min(origH - (int)((rect.Y - roi.Y) / scale), (int)(rect.Height / scale) + 2 * padding)
                    );

                    if (corrected.X >= 0 && corrected.Y >= 0 && corrected.Right <= origW && corrected.Bottom <= origH)
                        boxes.Add(corrected);
                }
           }
           return boxes;
        }
    }
}
