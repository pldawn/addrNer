package com.techwolf.addrNer;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import org.apache.commons.io.IOUtils;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class AddrTest {

	private Graph graph;
	private Session sess;
	
	public void loadModel( String[] fileArray ) throws FileNotFoundException, IOException
	{
		InputStream modelStream = getResource(fileArray[0]);
		
		graph = new Graph();
		byte[] graphBytes = IOUtils.toByteArray(modelStream);
		graph.importGraphDef(graphBytes);
		
		sess = new Session(graph);
	}
	
	private InputStream getResource( String resourceFileName ) throws FileNotFoundException {
		return this.getClass().getClassLoader().getResourceAsStream(resourceFileName);
	}
	
	public float[][] predictCosine( TitlePosInput input) {
		float[][] cosine = new float[input.word.length][70];
		
		try ( Tensor<Integer> word = Tensor.create(input.word, Integer.class)) 
		{
			Tensor<?> logit = sess.runner().feed("input_placeholder", word)
						.fetch("crf_sequences").run().get(0);
			logit.copyTo(cosine);
			logit.close();
		}
		
		return cosine;
	}
	
	public String convertToString( float[][] cosine) {
		StringBuffer strCos = new StringBuffer();
		strCos.append('{');
		for(int i=0; i<cosine.length; i+=1) {
			for(int j=0; j<cosine[i].length; j+=1) {
				if(j == 0) {
					strCos.append('{');
				}
				strCos.append(cosine[i][j]);
				if(j == cosine[i].length - 1) {
					strCos.append('}');
				}else {
					strCos.append(',');
				}
				if(i < cosine.length - 1) {
					strCos.append(',');
				}
			}
		}
		strCos.append('}');
		return strCos.toString();
	}
	
	public static void main(String[] args) throws FileNotFoundException, IOException {
		AddrTest agent = new AddrTest();
		agent.loadModel(new String[]{"addrNer.pb"});
		float[][] cosine = agent.predictCosine(new TitlePosInput());
		System.out.println(agent.convertToString(cosine));
	}

}
