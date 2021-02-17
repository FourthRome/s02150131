using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Forms;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using ImageRecognizer;

namespace GraphicalInterface
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public Task recognitionTask;
        public 

        public MainWindow()
        {
            InitializeComponent();
            recognitionTask = null;
        }

        public void OnClickStart(object sender, RoutedEventArgs e)
        {
            MnistRecognizer.CancelTokenSource = new CancellationTokenSource();
            //recognitionTask = Task.Run(async () => { await MnistRecognizer.ProcessImagesInDirectory(); });
        }

        public void OnClickAbort(object sender, RoutedEventArgs e)
        {
            MnistRecognizer.CancelTokenSource.Cancel();
        }

        public void OnClickChooseFolder(object sender, RoutedEventArgs e)
        {
            FolderBrowserDialog dlg = new FolderBrowserDialog();

            if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                  
            }
        }

        public void RecognitionCallback()
        {

        }

    }
}
