**Bonus part is implemeted in the code.**

### On running the code, the output answers the following three questions:

1.  *Do we really need all orders of a high-order Chebyshev/Legendre polynomial to stay below a predefined tolerance (1.E-6)?*

        No! The code finds which tailing high orders can be dropped, since max error for these special polynomials = sum(coeffiecients that are ignored)
    
2.  *How does Legendre fit compare with Chebyshev?*

        In general, the accuracy of our custom log function is comparable with both of them for small order of the polynomials. However, once we start fitting higher orders, Chebyshevs are much more well-behaved than Legendre. Legendre fits are susceptible to high errors randomly. So even though, the RMS error may be small, but the maximum error will be several orders of magnitude higher, as shown in the image `error_comparison.png`.