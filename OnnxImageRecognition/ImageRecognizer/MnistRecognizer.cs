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
using System.Diagnostics;

namespace ImageRecognizer
{
    // The struct to represent a single recognition result
    public struct RecognitionResult
    {
        // Model gives answers with confidence, so for each of the probabilities there is a separate entry
        public struct ResultEntry
        {
            public string Label { get; }
            public double Confidence { get; }
        
            public ResultEntry(string label, double confidence)
            {
                Label = label;
                Confidence = confidence;
            }
        }

        public string ImagePath { get; }
        public List<ResultEntry> ModelOutput { get; }
        public ResultEntry BestMatch { get; }


        // TODO: make RecognitionResult data class
        
        public RecognitionResult(string path, List<ResultEntry> modelOutput) 
        {
            ImagePath = path;
            ModelOutput = modelOutput;

            BestMatch = ModelOutput[0];  // It is guaranteed to be the first element now; however, there should be a better check
        }
    }


    public class MnistRecognizer
    {
        //---------------
        // Private fields
        //---------------
        static ConcurrentQueue<RecognitionResult> resultsQueue;
        static SemaphoreSlim newResults;  // To notify about new entries
        static SemaphoreSlim writePermission;  // To synchronize enqueue()
        static CancellationTokenSource cancelTokenSource;  // To stop processing new images
        static CancellationToken cancelToken;

        //-------------
        // Constructors
        //-------------
        static MnistRecognizer()
        {
            newResults = new SemaphoreSlim(0, 1);
            writePermission = new SemaphoreSlim(1, 1);
        }

        //-------------------------
        // Public static properties
        //-------------------------
        public static string[] ImageLabels { get => new string[] { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" }; }

        //----------------------
        // Public static methods
        //----------------------

        // Main interface of the library
        public static async Task ProcessImagesInDirectory(string path, Action<RecognitionResult> callback)
        {
            resultsQueue = new ConcurrentQueue<RecognitionResult>();
            cancelTokenSource = new CancellationTokenSource();
            cancelToken = cancelTokenSource.Token;
            //var processingCts = new CancellationTokenSource();  // Separate cancellation token for the result-processing task's cancellation
            var resultsProcessing = Task.Run(async () => { await ProcessRecognitionResults(callback); });  // TODO: remove a layer of abstraction here
            try
            {
                await TraverseDirectory(path);
            }
            catch (OperationCanceledException)
            {
                Trace.WriteLine($"[INFO] MnistRecognizer.ProcessImagesInDirectory: Catching TraverseDirectory OperationCanceledException");
            }
            
            cancelTokenSource.Cancel();
            await resultsProcessing;
        }

        // In case we need no more processing
        public static void StopProcessing()
        {
            if (cancelTokenSource != null)
            {
                try
                {
                    cancelTokenSource.Cancel();
                }
                catch (ObjectDisposedException)
                {
                    Trace.WriteLine($"[EXCEPTION] MnistRecognizer.Stop: Tried to stop recognition, but the CancellationTokenSource had already been disposed.");
                }
            }
        }

        //-----------------------
        // Private static methods
        //-----------------------

        // Method to run the callbacks if there are new results
        static async Task ProcessRecognitionResults(Action<RecognitionResult> callback)
        {
            while (!cancelToken.IsCancellationRequested)
            {
                try
                {
                    await newResults.WaitAsync(cancelToken);
                } catch (OperationCanceledException)
                {
                    Trace.WriteLine($"[INFO] MnistRecognizer.ProcessRecognitionResults: Interrupting NewResults.WaitAsync due to cancellation request");
                }
                await writePermission.WaitAsync();
                foreach (var entry in resultsQueue)
                {
                    try
                    {
                        callback(entry);
                    }
                    catch (Exception e)
                    {
                        Trace.WriteLine($"[CALLBACK EXCEPTION] MnistRecognizer.ProcessRecognitionResults: Exception thrown during result processing. Exception message: {e.Message}");
                    }
                }
                resultsQueue.Clear();
                writePermission.Release();
                await Task.Delay(100);
            }

            foreach (var entry in resultsQueue)
            {
                try
                {
                    callback(entry);
                }
                catch (Exception e)
                {
                    Trace.WriteLine($"[CALLBACK EXCEPTION] MnistRecognizer.ProcessRecognitionResults: Exception thrown during result processing. Exception message: {e.Message}");
                }
            }
        }

        // Method to start all the tasks
        static async Task TraverseDirectory(string path)
        {
            System.IO.DirectoryInfo dir;
            List<Task> routines = new List<Task>();
            try
            {
                // TODO: traverse directory in batches
                dir = new System.IO.DirectoryInfo(path);
                if (dir.Exists)
                {

                    foreach (FileInfo finfo in dir.GetFiles())
                    {
                        // Cancel adding new tasks, but all tasks that are started will still be completed
                        if (cancelToken.IsCancellationRequested)
                        {
                            break;
                        }

                        routines.Add(Task.Factory.StartNew(objFinfo =>
                        {
                            var fi = (FileInfo) objFinfo;
                            // TODO: Ask about capture here
                            Trace.WriteLine($"[INFO] Starting recognizing {fi} in a separate task");
                            Recognize(fi.FullName);
                        }, finfo, cancelToken));
                        Trace.WriteLine($"[INFO] MnistRecognizer.TraverseDirectory: put {finfo} to processing");

                    }
                }
                else
                {
                    Trace.WriteLine($"[INCORRECT PATH] MnistRecognizer.TraverseDirectory: Could not open \"{path}\"; the directory might not exist.");
                }
            }
            catch (TaskCanceledException) // TODO: Ask why this is leaking TaskCanceledException from the children tasks!
            {
                Trace.WriteLine($"[INFO] MnistRecognizer.TraverseDirectory: Cancellation from external source");
            }
            catch  // TODO: Is this catching cancellation exception too?
            {
                Trace.WriteLine($"[INCORRECT PATH] MnistRecognizer.TraverseDirectory: Could not get info about directory \"{path}\"; the directory might not exist.");

            }
            
            await Task.WhenAll(routines);
        }



        // Method to recognize a single image
        static void Recognize(string path, string modelPath="mnist-8.onnx")  // TODO: model path should be a parameter through the whole class lib
        {
            Image<Rgb24> image = null;
            try
            {
                image = Image.Load<Rgb24>(path);
            }
            catch (Exception e)
            {
                Trace.WriteLine($"[FILE ERROR] MnistRecognizer.Recognize: Could not read image \"{path}\": \n{e.Message}");
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
                // TODO: rewrite normalization properly
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
            catch  // TODO: make sure only this class of exceptions is caught here, and manage the others
            {
                Trace.WriteLine($"[FILE ERROR] MnistRecognizer.Recognize: Could not find \"{modelPath}\". Please follow the README.md and retry.");
                return;
            }

            // Getting results
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

            // Put the new result into the queue
            writePermission.Wait();
            resultsQueue.Enqueue(new RecognitionResult(path, res));
            try { newResults.Release(); } catch { }  // TODO: find a better solution for mutiple-releases-should-count-as-one problem
            writePermission.Release();
        }
    }
}
