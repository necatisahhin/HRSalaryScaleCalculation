
df = pd.read_csv("polynomial.csv",sep = ";")

plt.scatter(df['deneyim'],df['maas'])
plt.xlabel('Deneyim (yıl)')
plt.ylabel('Maaş')
plt.savefig('1.png', dpi=300)
plt.show()

reg = LinearRegression()
reg.fit(df[['deneyim']],df['maas'])

plt.xlabel('Deneyim (yıl)')
plt.ylabel('Maaş')

plt.scatter(df['deneyim'],df['maas'])   

xekseni = df['deneyim']
yekseni = reg.predict(df[['deneyim']])
plt.plot(xekseni, yekseni,color= "green", label = "linear regression")
plt.legend()
plt.show()

polynomial_regression = PolynomialFeatures(degree = 4)

x_polynomial = polynomial_regression.fit_transform(df[['deneyim']])


reg = LinearRegression()
reg.fit(x_polynomial,df['maas'])

y_head = reg.predict(x_polynomial)
plt.plot(df['deneyim'],y_head,color= "red",label = "polynomial regression")

plt.legend()

plt.scatter(df['deneyim'],df['maas'])   

plt.show()

x_polynomial1 = polynomial_regression.fit_transform([[4.5]])
reg.predict(x_polynomial1)
