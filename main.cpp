#include <iostream>
#include <fstream>

#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <cfloat>

#include <boost/date_time/gregorian/gregorian.hpp>


using namespace std;


pair<double,double> get_coef_for_smoothing(const vector<pair<boost::gregorian::date,double> >& vals)
{
    double min=1;
    double coef=0;
    for(int i=1;i<100;++i)
    {
        double cur_coef=double(i)/100;
        vector<double> fc;
        long double cso=0;
        fc.emplace_back(vals[0].second);
        for(size_t j=1;j<vals.size();++j)
        {
            fc.emplace_back(cur_coef*vals[j-1].second+(1-cur_coef)*fc[j-1]);
            cso+=(vals[j].second-fc[j])*(vals[j].second-fc[j])/((fc[j])*(fc[j]));
        }
        cso/=vals.size();
        if(cso<min) {min=cso;coef=cur_coef;}
    }
    return make_pair(coef,1-min);
}

pair<double,double> get_h_for_kernel_density(const vector<pair<boost::gregorian::date,double> >& vals)
{
    double min=DBL_MAX;
    double h=0.;
    for(int i=1;i<100;++i)
    {
        double cur_h=double(i)/100;
        double cv=0.;
        #pragma omp parallel for reduction(+:cv)
        for(size_t j=1;j<vals.size();++j)
        {
            long double numer=0.,denom=0.;
            for(size_t k=0;k<vals.size();++k)
            {
                if(k==j) continue;
                long double e=-(vals[j].second-vals[k].second)*(vals[j].second-vals[k].second)/(2*cur_h*cur_h);
                long double temp=exp(e);
                numer+=(sin(vals[k].second)+0.5)*temp;
                denom+=temp;
            }
            cv+=pow(vals[j].second-numer/denom,2);
        }
        cv/=vals.size();
        if(cv<min){min=cv;h=cur_h;}
    }
    return make_pair(0.3*h,min);
}

int main()
{
    map<string,vector<pair<boost::gregorian::date,double> > > values;
    std::ifstream in("torg3.txt");
    ofstream out("results.csv");
    cout << fixed << setprecision(6);
    out << fixed << setprecision(6);
    boost::gregorian::date_input_facet* facet(new boost::gregorian::date_input_facet("%d.%m.%Y"));
    in.imbue(std::locale(in.getloc(), facet));
    while(in.good())
    {
        string company="";
        in >> company;
        boost::gregorian::date price_date;
        in >> price_date;
        double price =0.;
        in >> price;
        auto place = values.find(company);
        if(place==values.end()) place=(values.emplace(company,vector<pair<boost::gregorian::date,double> >())).first;
        place->second.emplace_back(price_date,price);
    }
    values.erase("");
    cout <<"Complete!\nNumber of companies: " << values.size() << '\n';
    out << "Company;Alpha;h*\n";
    for(auto& el:values)
    {
        cout << el.first << '\n';
        out << el.first << ';';
        pair<double,double> esc=get_coef_for_smoothing(el.second);
        cout << "esc: " << esc.first << " prob = " << esc.second << '\n';
        out << esc.first << ';';
        pair<double,double> h=get_h_for_kernel_density(el.second);
        cout << "h*: " << h.first << "prob= " << h.second << '\n';
        out << h.first << '\n';

    }
    delete facet;
    return 0;
}

