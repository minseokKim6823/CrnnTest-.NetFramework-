using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace CrnnTest
{
    public partial class MainForm : Form
    {
        private OcrPipeline pipeline;
        private string imagePath;
        private Mat inputImage;
        private Button RUN;

        public MainForm()
        {
            InitializeComponent();
            pipeline = new OcrPipeline();
        }

        private void btnOpen_Click(object sender, EventArgs e)
        {
            using (var dlg = new OpenFileDialog())
            {
                dlg.Filter = "Image Files|*.png;*.jpg;*.jpeg;*.bmp";
                if (dlg.ShowDialog() == DialogResult.OK)
                {
                    imagePath = dlg.FileName;
                    pictureBox.Image = new Bitmap(imagePath);
                }
            }
        }

        private void InitializeComponent()
        {
            this.pictureBox = new System.Windows.Forms.PictureBox();
            this.resultTextBox = new System.Windows.Forms.TextBox();
            this.open = new System.Windows.Forms.Button();
            this.RUN = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox)).BeginInit();
            this.SuspendLayout();

            this.pictureBox.Location = new System.Drawing.Point(12, 12);
            this.pictureBox.Name = "pictureBox";
            this.pictureBox.Size = new System.Drawing.Size(1015, 451);
            this.pictureBox.TabIndex = 0;
            this.pictureBox.TabStop = false;
            this.pictureBox.Click += new System.EventHandler(this.pictureBox_Click);
            // 
            // resultTextBox
            // 
            this.resultTextBox.Location = new System.Drawing.Point(361, 489);
            this.resultTextBox.Multiline = true;
            this.resultTextBox.Name = "resultTextBox";
            this.resultTextBox.Size = new System.Drawing.Size(323, 140);
            this.resultTextBox.TabIndex = 1;
            this.resultTextBox.TextChanged += new System.EventHandler(this.resultTextBox_TextChanged);
            // 
            // open
            // 
            this.open.FlatAppearance.BorderSize = 5;
            this.open.Font = new System.Drawing.Font("굴림", 18F, System.Drawing.FontStyle.Bold);
            this.open.ForeColor = System.Drawing.Color.Black;
            this.open.Location = new System.Drawing.Point(790, 577);
            this.open.Name = "open";
            this.open.Size = new System.Drawing.Size(176, 52);
            this.open.TabIndex = 2;
            this.open.Text = "UPLOAD";
            this.open.UseVisualStyleBackColor = true;
            this.open.Click += new System.EventHandler(this.button1_Click);
            // 
            // RUN
            // 
            this.RUN.FlatAppearance.BorderSize = 5;
            this.RUN.Font = new System.Drawing.Font("굴림", 18F, System.Drawing.FontStyle.Bold);
            this.RUN.ForeColor = System.Drawing.Color.Black;
            this.RUN.Location = new System.Drawing.Point(790, 489);
            this.RUN.Name = "RUN";
            this.RUN.Size = new System.Drawing.Size(176, 52);
            this.RUN.TabIndex = 4;
            this.RUN.Text = "RUN";
            this.RUN.UseVisualStyleBackColor = true;
            this.RUN.Click += new System.EventHandler(this.RUN_Click);
            // 
            // MainForm
            // 
            this.ClientSize = new System.Drawing.Size(1039, 641);
            this.Controls.Add(this.RUN);
            this.Controls.Add(this.open);
            this.Controls.Add(this.resultTextBox);
            this.Controls.Add(this.pictureBox);
            this.Name = "MainForm";
            this.Load += new System.EventHandler(this.MainForm_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        private PictureBox pictureBox;
        private TextBox resultTextBox;

        private void MainForm_Load(object sender, EventArgs e)
        {

        }

        private Button open;

        private void button1_Click(object sender, EventArgs e)
        {
            using(var dlg = new OpenFileDialog())
            {
                dlg.Filter = "이미지 파일|*.png;*.jpg;*.jpeg;*.bmp";
                dlg.Title = " 이미지 파일 열기";
                dlg.Multiselect = false;

                if(dlg.ShowDialog() == DialogResult.OK)
                {
                    string filePath = dlg.FileName;

                    try
                    {
                        inputImage = CvInvoke.Imread(filePath, ImreadModes.Color);

                        if (inputImage == null || inputImage.IsEmpty)
                        {
                            MessageBox.Show("이미지를 불러오는 데 실패했습니다.");
                            return;
                        }
                        pictureBox.Image = inputImage.ToBitmap();
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show("이미지 로딩 오류: " + ex.Message);
                    }
                }
            }
        }

        private void RUN_Click(object sender, EventArgs e)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            if (inputImage == null || inputImage.IsEmpty)
            {
                MessageBox.Show("이미지를 먼저 업로드해주세요.");
                return;
            }

            // 복제 이미지
            Mat imgWithBoxes = inputImage.Clone();

            // 1. Detection: DBNet으로 텍스트 박스 탐지
            string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "en_PP-OCRv3_det_infer.onnx");
            DbNetDetector dbnet = new DbNetDetector(modelPath);
            List<Rectangle> detectedBoxes = dbnet.Detect(inputImage);

            //foreach (var rect in detectedBoxes)
            //{
            //    CvInvoke.Rectangle(imgWithBoxes, rect, new MCvScalar(0, 0, 255), 2);  // 빨간색 박스
            //}

            // 2. OCR 파이프라인
            var result = pipeline.Run(inputImage, detectedBoxes);  // result.BoxImg도 있음

            // 3. OCR 고정된 박스도 표시 (serial, amount, micr 영역)
            if (result.BoxImg != null)
            {
                Bitmap baseBitmap = imgWithBoxes.ToBitmap();         // 원본 + detection 박스
                Bitmap ocrBitmap = result.BoxImg.ToBitmap();         // OCR 박스 그려진 것

                using (Graphics g = Graphics.FromImage(baseBitmap))
                {
                    g.DrawImage(ocrBitmap, 0, 0); // 겹쳐서 그림
                }

                pictureBox.Image = baseBitmap;
            }
            else
            {
                pictureBox.Image = imgWithBoxes.ToBitmap();
            }

            // 4. 화면에 출력
            resultTextBox.Text = result.ToString();

            stopwatch.Stop();
            resultTextBox.Text = result.ToString() +$"\n\n🕒 실행 시간: {stopwatch.ElapsedMilliseconds} ms";
         }

        private void resultTextBox_TextChanged(object sender, EventArgs e)
        {

        }

        private void pictureBox_Click(object sender, EventArgs e)
        {

        }
    }
}