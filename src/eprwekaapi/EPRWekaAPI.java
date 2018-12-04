/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eprwekaapi;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
/**
 *
 * @author LENOVO
 */
public class EPRWekaAPI {

    /**
     * @param args the command line arguments
     */
     public static String[] J48Decisions=new String[1];
    public static String[] NNDecisions=new String[1];
    public static String[] SVMDecisions=new String[1];
    public static int i=0;
        public static void writeCSV(String container) throws Exception
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(container));
       
            writer.write(J48Decisions[0]+","+NNDecisions[0]+","+SVMDecisions[0]);
            writer.newLine();
        
        writer.close();
    }
    
    public static void J48Test (Instances test,String model)throws Exception
    {
       System.out.println("Interpreting J48 Model");
       J48 j48;
      j48= (J48)(SerializationHelper.read(new FileInputStream(model)));
      Evaluation eval=new Evaluation(test);
      eval.useNoPriors();
      System.out.println(eval.toMatrixString());
     
          eval.evaluateModelOnce(j48, test.get(i));
          if(eval.confusionMatrix()[0][0]==1||eval.confusionMatrix()[1][0]==1||eval.confusionMatrix()[2][0]==1)
          {
              J48Decisions[0]="High";
              System.out.println("High");
          }
          else if(eval.confusionMatrix()[0][1]==1||eval.confusionMatrix()[1][1]==1||eval.confusionMatrix()[2][1]==1)
          {
               J48Decisions[0]="Low";
              System.out.println("Low");
          }
          else
          {
              J48Decisions[0]="Medium";
              System.out.println("Medium");
          }
           for(int a=0;a<3;a++)
         {
             for(int b=0;b<3;b++)
             {
                 System.out.print(eval.confusionMatrix()[a][b]+" ");
             }
             System.out.println(" ");
         }
     
    }
    public static void NNTest (Instances test,String model)throws Exception
    {
         System.out.println("Interpreting Neural Network Model");
       MultilayerPerceptron mp;
      mp= (MultilayerPerceptron)(SerializationHelper.read(new FileInputStream(model)));
     
      Evaluation eval=new Evaluation(test);
      eval.useNoPriors();
      System.out.println(eval.toMatrixString());
    
          eval.evaluateModelOnce(mp, test.get(i));
            if(eval.confusionMatrix()[0][0]==1||eval.confusionMatrix()[1][0]==1||eval.confusionMatrix()[2][0]==1)
          {
              NNDecisions[0]="High";
              System.out.println("High");
          }
          else if(eval.confusionMatrix()[0][1]==1||eval.confusionMatrix()[1][1]==1||eval.confusionMatrix()[2][1]==1)
          {
               NNDecisions[0]="Low";
              System.out.println("Low");
          }
          else
          {
              NNDecisions[0]="Medium";
              System.out.println("Medium");
          }
            for(int a=0;a<3;a++)
         {
             for(int b=0;b<3;b++)
             {
                 System.out.print(eval.confusionMatrix()[a][b]+" ");
             }
             System.out.println(" ");
         }
    }
    public static void SVMTest (Instances test,String model)throws Exception
    {
         System.out.println("Interpreting Support Vector Machine Model");
       SMO smo;
      smo= (SMO)(SerializationHelper.read(new FileInputStream(model)));
      Evaluation eval=new Evaluation(test);
      eval.useNoPriors();
      System.out.println(eval.toMatrixString());
      
          eval.evaluateModelOnce(smo, test.get(i));
           if(eval.confusionMatrix()[0][0]==1||eval.confusionMatrix()[1][0]==1||eval.confusionMatrix()[2][0]==1)
          {
              SVMDecisions[0]="High";
              System.out.println("High");
          }
          else if(eval.confusionMatrix()[0][1]==1||eval.confusionMatrix()[1][1]==1||eval.confusionMatrix()[2][1]==1)
          {
               SVMDecisions[0]="Low";
              System.out.println("Low");
          }
          else
          {
              SVMDecisions[0]="Medium";
              System.out.println("Medium");
          }
         for(int a=0;a<3;a++)
         {
             for(int b=0;b<3;b++)
             {
                 System.out.print(eval.confusionMatrix()[a][b]+" ");
             }
             System.out.println(" ");
         }
    }

    public static void main(String[] args) throws Exception{
        System.out.println("Interpreting Models");
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(args[0]));
        Instances data = loader.getDataSet();
        data.setClassIndex(39);
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(args[1]));
        saver.writeBatch();
        J48Test(data,args[2]);
        NNTest(data,args[3]);
        SVMTest(data,args[4]);
        writeCSV(args[5]);
    }
    
}
