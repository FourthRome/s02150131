using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Threading.Tasks;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Threading;

namespace ImageRecognizer
{
    public struct RecognitionResult
    {
        public struct ResultEntry
        {
            public string Label { get; set; }
            public double Confidence { get; set; }
        
            public ResultEntry(string label, double confidence)
            {
                Label = label;
                Confidence = confidence;
            }
        }

        public string ImagePath { get; set; }
        public List<ResultEntry> ModelOutput { get; }

        // TODO: make RecognitionResult data class
        
        public RecognitionResult(string path, List<ResultEntry> modelOutput) 
        {
            ImagePath = path;
            ModelOutput = modelOutput;
        }
    }


    public class MnistRecognizer
    {
        public static ConcurrentQueue<RecognitionResult> ResultsQueue;
        public static SemaphoreSlim NewResults;
        
        private static SemaphoreSlim WritePermission;

        static MnistRecognizer()
        {
            ResultsQueue = new ConcurrentQueue<RecognitionResult>();
            NewResults = new SemaphoreSlim(0, 1);
            WritePermission = new SemaphoreSlim(1, 1);
        }

        public static async Task TraverseDirectory(string path)
        {
            System.IO.DirectoryInfo dir;
            List<Task> routines = new List<Task>();
            try
            {
                dir = new System.IO.DirectoryInfo(path);
                if (dir.Exists)
                {
                    foreach (FileInfo fi in dir.GetFiles())
                    {
                        routines.Add(Task.Factory.StartNew(() =>
                        {
                            Recognize(fi.FullName);
                        }));
                    }
                }
                else
                {
                    System.Diagnostics.Trace.WriteLine($"[INCORRECT PATH] MnistRecognizer.TraverseDirectory: Could not open \"{path}\"; the directory might not exist.");
                }
            }
            catch
            {
                System.Diagnostics.Trace.WriteLine($"[INCORRECT PATH] MnistRecognizer.TraverseDirectory: Could not get info about directory \"{path}\"; the directory might not exist.");
            }
            
            await Task.WhenAll(routines);
        }

        public static void Recognize(string path, string modelPath="mnist-8.onnx")
        {
            Image<Rgb24> image = null;
            try
            {
                image = Image.Load<Rgb24>(path);
            }
            catch (Exception e)
            {
                // TODO: proper logging
                System.Diagnostics.Trace.WriteLine($"[FILE ERROR] MnistRecognizer.Recognize: Could not read image \"{path}\": \n{e.Message}");
                return;
            }

            // These come from the model's requirements
            const int TargetWidth = 28;
            const int TargetHeight = 28;

            // Changing image size and making it grayscale
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(TargetWidth, TargetHeight),
                    Mode = ResizeMode.Crop
                });
                x.Grayscale();
            });

            // Converting to a tensor and normalization
            var input = new DenseTensor<float>(new[] { 1, 1, TargetHeight, TargetWidth });
            for (int y = 0; y < TargetHeight; y++)
            {
                Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);
                for (int x = 0; x < TargetWidth; ++x)
                {
                    input[0, 0, y, x] = pixelSpan[x].R / 255f;
                }
            }

            // Preparing neural net's inputs. 'Input3' name is used in the model itself
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("Input3", input)
            };

            // Inference
            InferenceSession session;
            try
            {
                session = new InferenceSession(modelPath);
            }
            catch
            {
                System.Diagnostics.Trace.WriteLine($"[FILE ERROR] MnistRecognizer.Recognize: Could not find \"{modelPath}\". Please follow the README.md and retry.");
                return;
            }

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // Applying softmax
            var output = results.First().AsEnumerable<float>().ToArray();
            var sum = output.Sum(x => (float)Math.Exp(x));
            var softmax = output.Select(x => (float)Math.Exp(x) / sum);

            var res = softmax
                .Select((x, i) => new RecognitionResult.ResultEntry(label : i.ToString(), confidence : x))
                .OrderByDescending(x => x.Confidence)
                .Take(10)
                .ToList();

            WritePermission.Wait();
            ResultsQueue.Enqueue(new RecognitionResult(path, res));
            try { NewResults.Release(); } catch { }
            WritePermission.Release();

            Console.WriteLine("Recognition finished");
        }
    }
}
