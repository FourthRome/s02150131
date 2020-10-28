using System;
using System.Threading;
using System.Threading.Tasks;
using ImageRecognizer;

namespace ConsoleInterface
{
    class Program
    {
        //---------------
        // Static methods
        //---------------
        static async Task Main(string[] args)
        {
            Console.WriteLine("Type the full path to the folder, and image recognition will begin:");
            string path = Console.ReadLine();

            // Run task to wait for cancellation from keyboard
            var cancellationTask = Task.Run(() => { CancellationWaiter(MnistRecognizer.CancelTokenSource); });

            // Start processing
            await MnistRecognizer.ProcessImagesInDirectory(path, PrintResultEntry);
        }

        static void PrintResultEntry(RecognitionResult entry)
        {
            Console.WriteLine($"Results for {entry.ImagePath}:");

            foreach (var output in entry.ModelOutput)
            {
                Console.WriteLine($"{output.Label} with confidence {output.Confidence}");
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
