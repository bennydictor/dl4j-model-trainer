// In the tk.bennydictor.chorme package is my futile attempt
// To parse CHORME by hand. Please ignore it.
// We probably should use compvis to parse CHORME and output images in png or something.
package tk.bennydictor;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;

import javax.xml.parsers.ParserConfigurationException;
import java.io.IOException;
import java.util.List;

public class Dl4jModelTrainerMain {
    // Do we want to evaluate the network
    // If EVAL is false, the all dataset is for trainings
    // Otherwise, it's 80% training 20% testing
    private static final boolean EVAL = true;
    private static final int EPOCHS = 15;
    public static void main(String[] args) throws IOException, ParserConfigurationException {
        // Get the configuration
        MultiLayerNetwork model = ModelLayout.get();
        // Get the dataset
        DataSet dataSet = CROHME.get();

        DataSet train, test;
        if (EVAL) {
            // Split the dataset randomly 80% train 20% test
            dataSet.shuffle();
            SplitTestAndTrain split = dataSet.splitTestAndTrain(.97);
            train = split.getTrain();
            test = split.getTest();
        } else {
            train = dataSet;
        }

        for (int i = 0; i < EPOCHS; ++i) {
            train.shuffle();
            // Batching is very important
            // If you try to model.fit the entire dataset,
            // java will run out of memory
            List<DataSet> batches = train.batchBy(128);
            System.out.print("Epoch " + (i + 1) + "/" + EPOCHS + " ");
            for (DataSet d : batches) {
                System.out.print('.');
                model.fit(d);
            }
            System.out.println();
            if (EVAL) {
                Evaluation eval = new Evaluation();
                INDArray guesses = model.output(test.getFeatureMatrix());
                eval.eval(test.getLabels(), guesses);
                // Print pretty stats
                System.out.println(eval.stats());
            }
        }

        // Save the model
        ModelSerializer.writeModel(model, "model.zip", false);
    }
}
