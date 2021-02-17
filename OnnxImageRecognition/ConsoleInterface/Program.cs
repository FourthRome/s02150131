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
            var cancellationTask = Task.Run(async () => { await CancellationWaiter(); });  // Isn't there too many layers of abstraction?

            // Start processing
            await MnistRecognizer.ProcessImagesInDirectory(path, PrintResultEntry);
        }

        // A callback for every new result
        static void PrintResultEntry(RecognitionResult entry)
        {
            Console.WriteLine($"Results for {entry.ImagePath}:");

            foreach (var output in entry.ModelOutput)
            {
                Console.WriteLine($"{output.Label} with confidence {output.Confidence}");
            }
        }

        // Wait for a key combination to stop spawning new tasks
        static async Task CancellationWaiter(CancellationTokenSource cts = null)
        {
            if (cts == null)  // If cts is not provided externally, the waiter will kill itself
            {
                cts = new CancellationTokenSource();
            }

            while (!cts.Token.IsCancellationRequested)
            {
                if (!Console.KeyAvailable)
                {
                    await Task.Delay(100);
                }
                else
                {
                    var key = Console.ReadKey(intercept: true);
                    if (key.Key == ConsoleKey.C)
                    {
                        cts.Cancel();
                        Console.WriteLine("Termination is requested...");
                        MnistRecognizer.StopProcessing();
                        return;
                    }
                }
            }
        }
    }
}
