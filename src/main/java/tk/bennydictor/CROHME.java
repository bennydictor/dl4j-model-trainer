package tk.bennydictor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.*;
import java.awt.image.DataBufferInt;
import java.awt.image.Raster;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;



import java.awt.image.BufferedImage;
import java.io.*;
import javax.imageio.ImageIO;
import com.sun.org.apache.xerces.internal.impl.dv.util.Base64;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;

import static jdk.nashorn.internal.objects.ArrayBufferView.length;

public class CROHME {
     static int folderSize(File directory) {
        int len = 0;
        for (File file : directory.listFiles()) {
            if (file.isFile())
                len += 1;
            else
                len += folderSize(file);
        }
        return len;
    }

    static DataSet get() {
        File symbolDir = new File("data");          //symbolDir -- это папка с папками-названиями символов
        int size = folderSize(symbolDir);
        int offset = 0;
        int num_images = 75; // 75
        INDArray inputs = Nd4j.zeros(size, 50 * 50);
        INDArray outputs = Nd4j.zeros(size, num_images);
        PrintWriter charMap = null;
        try {
            charMap = new PrintWriter(new FileWriter(new File("charmap.txt")));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        File[] listOfFiles = symbolDir.listFiles(); //listOfFiles -- это список файлов-папок-названий символов
        for (int countSym = 0; countSym < num_images; ++countSym) { //countSym - счетчик символов
            String NameSym = listOfFiles[countSym].getName();  //NameSym -- имя папки-названия символа номер countSym
            System.out.println(countSym + 1 + " symbol: " + folderSize(listOfFiles[countSym]) + " \t" + NameSym);
            charMap.println(NameSym.length() > 1 ? "\\" + NameSym + "{}" : NameSym);
            charMap.flush();
            for (int countSymSample = 1; countSymSample <= folderSize(listOfFiles[countSym]); ++countSymSample) { //countSymSample -- счетчик экземпляра символа
                try (FileInputStream fileStream = new FileInputStream("data/" + NameSym + "/" + countSymSample + ".png")) {
                    BufferedImage image = ImageIO.read(fileStream);
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    ImageIO.write(image, "png", baos);
                    float[][] pixels = new float[50][50];

                    for (int i = 0; i < 50; ++i) {
                        for (int j = 0; j < 50; ++j) {
                            int pixel = image.getRGB(i, j);
                            int red= (pixel>>16)&255;
                            int green= (pixel>>8)&255;
                            int blue= (pixel)&255;
                            pixels[j][i] = 1f - ((red + green + blue) / 255f / 3f);
                        }
                    }
                    for (int i = 0; i < 50; ++i) {
                        for (int j = 0; j < 50; ++j) {
                            inputs.putScalar(offset, i * 50 + j, pixels[i][j]);
                        }
                    }
                    outputs.putScalar(offset, countSym, 1f);
                    offset += 1;
                } catch (IOException e) {
                    throw new Error(e);
                }
            }
        }
        return new DataSet(inputs, outputs);
    }
}

