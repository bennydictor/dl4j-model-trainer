package tk.bennydictor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;


class MNIST {
    static DataSet get() {

        int samples;
        {
            File digitZero = new File("data1/data0");
            samples = (int) digitZero.length() / (28 * 28);
            System.out.println("samples: " + samples);
        }
        INDArray inputs = Nd4j.zeros(10 * samples, 28 * 28);
        INDArray outputs = Nd4j.zeros(10 * samples, 10);
        for (int digit = 0; digit < 10; ++digit) {
            try (FileInputStream file = new FileInputStream(new File("data1/data" + digit))) {
                byte[] bytes = new byte[28 * 28];
                for (int sample = 0; file.read(bytes) == 28 * 28; ++sample) {
                    for (int i = 0; i < 28 * 28; ++i) {
                        inputs.putScalar(digit * samples + sample, i, Byte.toUnsignedInt(bytes[i]) / 255f);
                    }
                    outputs.putScalar(digit * samples + sample, digit, 1f);
                }
            } catch (IOException e) {
                throw new Error(e);
            }
        }
        return new DataSet(inputs, outputs);
    }
}
