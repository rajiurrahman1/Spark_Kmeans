// Author: Rajiur Rahman ( rajiurrahman.bd@gmail.com )
// Department of Computer Science, Wayne State University

// For installation and running of the codes, please find the appropriate tutorial from the following link
//      http://dmkd.cs.wayne.edu/TUTORIAL/Bigdata/Codes/


package org.sparkexample;

import com.google.common.collect.Iterators;
import org.apache.spark.SparkConf;
import scala.Tuple2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.util.Vector;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

public class TestKMeans_New {
    private static final Pattern SPACE = Pattern.compile(" ");
    static Vector parseVector(String line) {
        String[] splits = SPACE.split(line);
        double[] data = new double[splits.length];
        int i = 0;
        for (String s : splits) {
            data[i] = Double.parseDouble(s);
            i++;
        }
        splits = null;
        return new Vector(data);
    }

    public static class ReadData implements Function<String, Vector>{
        public Vector call(String s){
            return parseVector(s);
        }
    }

    static int closestPoint(Vector p, List<Vector> centers) {
        int bestIndex = 0;
        double closest = Double.POSITIVE_INFINITY;
        for (int i = 0; i < centers.size(); i++) {
            double tempDist = p.squaredDist(centers.get(i));
            if (tempDist < closest) {
                closest = tempDist;
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    static Vector average(List<Vector> ps) {
        int numVectors = ps.size();
        Vector out = new Vector(ps.get(0).elements());
        for (int i = 0; i < numVectors; i++) {
            out.addInPlace(ps.get(i));
        }
        return out.divide(numVectors);
    }

    private static double[] sumDataPoints(double[] sumPoints, double[] elements, int dataDimension) {
        double[] sumDataPoints = new double[dataDimension];
        Arrays.fill(sumDataPoints, 0.0);
        /*if (sumDataPoints.length == dataDimension && elements.length == dataDimension){
            for (int i = 0; i < dataDimension; i++) {
                sumDataPoints[i] = sumPoints[i] + elements[i];
            }
            return sumDataPoints;
        }*/

        if(elements.length < dataDimension){
            for(int i=0; i<elements.length; i++){
                sumDataPoints[i] = sumPoints[i] + elements[i];
            }
        }
        else{
            for(int i=0; i<dataDimension; i++){
                sumDataPoints[i] = sumPoints[i] + elements[i];
            }
        }
        return sumDataPoints;

    }

    private static double[] calculateAverage(double[] sumPoints, int numVectorsInCluster, int dataDimension) {
        double[] averagedDataPoints = new double[dataDimension];
        Arrays.fill(averagedDataPoints, 0.0);
        for(int i=0; i<dataDimension; i++){
            averagedDataPoints[i] = sumPoints[i]/(double)numVectorsInCluster;
        }
        return averagedDataPoints;
    }

    static Vector average1(Iterable<Vector> vectors, int dataDimension){
        int numVectorsInCluster = 0;
        double[] sumPoints = new double[dataDimension];
        Arrays.fill(sumPoints, 0.0);
        for(Vector v:vectors){
            numVectorsInCluster ++;
            sumPoints = sumDataPoints(sumPoints, v.elements(), dataDimension);
        }
        double[] avgPoints = calculateAverage(sumPoints, numVectorsInCluster, dataDimension);
        Vector returnVector = new Vector(avgPoints);
        sumPoints = null;
        avgPoints = null;

        return returnVector;
    }

    public static void main (String[] args){
        long startTime = System.currentTimeMillis();
        final int dataDimension = Integer.parseInt(args[2]);
        int numberOfIteration = Integer.parseInt(args[3]);
        int numberOfClusters = Integer.parseInt(args[4]);

        System.out.println("\n\nData Dimension: "+dataDimension+"\n\n\n\n\n");

        SparkConf sparkConf = new SparkConf().setAppName("TestKMeans");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        String filePath = args[0];
        JavaRDD<String> lines = sc.textFile(filePath);
        JavaRDD<Vector> data = lines.map(new ReadData()).cache();
        long instanceCount = data.count();
        lines.unpersist();
        lines = null;
        System.out.println("\n\n\n\nData Read Complete \n\n\n\nNumber of instances: " + instanceCount+"\n\n\n\n");

        String centroidsFielPath = (args[1]);
        JavaRDD<String> lines1 = sc.textFile(filePath);
        JavaRDD<Vector> centroidsRDD = lines1.map(new ReadData()).cache();

//        final List<Vector> centroids = data.takeSample(false, numberOfClusters);
        final List<Vector> centroids = centroidsRDD.collect();

        for(int iteration=0; iteration<numberOfIteration; iteration++){
            System.out.println("\n\n\nIteration: " + iteration + "\n\n");

            JavaPairRDD<Integer, Vector> closest = data.mapToPair(
                    new PairFunction<Vector, Integer, Vector>() {
                        @Override
                        public Tuple2<Integer, Vector> call(Vector vector) {
                            return new Tuple2<Integer, Vector>(
                                    closestPoint(vector, centroids), vector);
                        }
                    }
            );
            JavaPairRDD<Integer, Iterable<Vector>> pointsGroup = closest.groupByKey();
            Map<Integer, Vector> newCentroids = pointsGroup.mapValues(
                new Function<Iterable<Vector>, Vector>() {
                    public Vector call(Iterable<Vector> ps) throws Exception {
                        return average1(ps, dataDimension);
                    }
            }).collectAsMap();

            for (Map.Entry<Integer, Vector> t: newCentroids.entrySet()) {
                centroids.set(t.getKey(), t.getValue());
            }

        }

        long endTime   = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("\n\n\nEnd of Spark Program\nTotal time taken: "+totalTime+"\n\n\n");
    }
}
