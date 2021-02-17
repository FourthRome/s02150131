using ImageRecognizer;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace GraphicalInterface
{
    class ViewModel
    {
        public string[] ImageLabels { get; private set; }

        public Task RecognitionTask { get; set; }

        public List<RecognitionResult> RecognitionResults { get; set; }

        public ViewModel()
        {
            ImageLabels = MnistRecognizer.ImageLabels;
            RecognitionTask = null;
            RecognitionResults = new List<RecognitionResult>();
        }
    }
}
