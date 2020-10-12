using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ImageRecognizer;

namespace ConsoleInterface
{
    class Program
    {
        static readonly TaskStatus[] recognitionFinishedMarkers = {
            TaskStatus.Faulted,
            TaskStatus.Canceled,
            TaskStatus.RanToCompletion,
        };
        static async Task Main(string[] args)
        {
            Console.WriteLine("Type the full path to the folder, and image recognition will begin:");
            string path = Console.ReadLine();

            // Create cancellation items
            CancellationTokenSource cts = new CancellationTokenSource();
            CancellationToken cancelToken = cts.Token;

            // Run task to wait for cancellation from keyboard
            var cancellationTask = Task.Run(() => { CancellationWaiter(cts); }, cancelToken);

            // Start processing
            var traversing = MnistRecognizer.TraverseDirectory(path, cancelToken);

            // Print new items in results' queue
            while (!recognitionFinishedMarkers.Contains(traversing.Status))
            {
                await MnistRecognizer.NewResults.WaitAsync();
                await MnistRecognizer.WritePermission.WaitAsync();
                PrintQueue();
                MnistRecognizer.ResultsQueue.Clear();
                MnistRecognizer.WritePermission.Release();
                await Task.Delay(100);
            }

            PrintQueue();
        }

        static void PrintQueue()
        {
            foreach (var entry in MnistRecognizer.ResultsQueue)
            {
                Console.WriteLine($"Results for {entry.ImagePath}:");

                foreach (var output in entry.ModelOutput)
                {
                    Console.WriteLine($"{output.Label} with confidence {output.Confidence}");
                }
            }
        }

        static void CancellationWaiter(CancellationTokenSource cts)
        {
            while (!cts.Token.IsCancellationRequested)
            {
                if (!Console.KeyAvailable)
                {
                    Task.Delay(100);
                }
                else
                {
                    var key = Console.ReadKey(intercept: true);
                    if (key.Key == ConsoleKey.C)
                    {
                        cts.Cancel();
                        Console.WriteLine("Termination is requested...");
                        return;
                    }
                }
            }
        }
    }
}
