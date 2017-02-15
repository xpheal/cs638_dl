main: Lab1.java
	javac Lab1.java

protein:
	java Lab1 data/protein-secondary-structure.train

clean:
	rm -f *.class