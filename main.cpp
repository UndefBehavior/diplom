#include <iostream>
#include <fstream>

#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <cfloat>

#include <boost/date_time/gregorian/gregorian.hpp>


using namespace std;

//Y`(t+1)=alpha*Y(t)+(1-alpha)*Y`(t)
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

double get_h_for_kernel_density(const vector<pair<boost::gregorian::date,double> >& vals)
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
    return 0.3*h;
}

//нулевой - либо минимум, либо ничего
vector<pair<boost::gregorian::date,double> > get_min_max_for_exp_method(const vector<pair<boost::gregorian::date,double> >& vals, double alpha)
{
    vector<double> func;
    vector<pair<boost::gregorian::date,double> > res;
    func.emplace_back(vals[0].second);
    for(size_t j=1;j<vals.size();++j)
        func.emplace_back(alpha*vals[j-1].second+(1.-alpha)*func[j-1]);
    bool flag;//0 - последним внесен минимум; 1 - последним внесен максимум.
    if(func[1]<func[2]) {res.emplace_back(vals[1].first,func[1]);flag=0;}
    else
    {
        res.emplace_back(boost::gregorian::date(),0.);
        res.emplace_back(vals[1].first,func[1]);flag=1;
    }
    for(size_t i=2;i<func.size()-1;++i)
    {
        if(flag)
        {
            if(!((func[i-1]>func[i])&&(func[i+1]>func[i]))) continue;
            flag^=1;
            res.emplace_back(vals[i].first,func[i]);
        }
        else
        {
            if(!((func[i-1]<func[i])&&(func[i+1]<func[i]))) continue;
            flag^=1;
            res.emplace_back(vals[i].first,func[i]);
        }
    }
    if(((func[func.size()-2]<func[func.size()-1])&&(!flag))||((func[func.size()-2]>func[func.size()-1])&&flag)) res.emplace_back(vals[func.size()-1].first,func[func.size()-1]);
    return res;
}

vector<pair<boost::gregorian::date,double> > get_min_max_for_kernel_density(const vector<pair<boost::gregorian::date,double> >& vals, double h)
{
    vector<double> func(vals.size());
    vector<pair<boost::gregorian::date,double> > res;
#pragma omp parallel for
    for(size_t j=0;j<vals.size();++j)
    {
        long double numer=0.,denom=0.;
        for(size_t k=0;k<vals.size();++k)
        {
            if(k==j) continue;
            long double e=-(vals[j].second-vals[k].second)*(vals[j].second-vals[k].second)/(2*h*h);
            long double temp=exp(e);
            numer+=(sin(vals[k].second)+0.5)*temp;
            denom+=temp;
        }
        func[j]=numer/denom;
    }
    bool flag;//0 - последним внесен минимум; 1 - последним внесен максимум.
    if(func[0]<func[1]) {res.emplace_back(vals[0].first,func[0]);flag=0;}
    else
    {
        res.emplace_back(boost::gregorian::date(),0.);
        res.emplace_back(vals[0].first,func[0]);flag=1;
    }
    for(size_t i=1;i<func.size()-1;++i)
    {
        if(flag)
        {
            if(!((func[i-1]>func[i])&&(func[i+1]>func[i]))) continue;
            flag^=1;
            res.emplace_back(vals[i].first,func[i]);
        }
        else
        {
            if(!((func[i-1]<func[i])&&(func[i+1]<func[i]))) continue;
            flag^=1;
            res.emplace_back(vals[i].first,func[i]);
        }
    }
    if(((func[func.size()-2]<func[func.size()-1])&&(!flag))||((func[func.size()-2]>func[func.size()-1])&&flag)) res.emplace_back(vals[func.size()-1].first,func[func.size()-1]);
    return res;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_hs_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    for(size_t i=1;i<min_maxes.size()-4;i+=2)
    {
        if(!(min_maxes[i+2].second>min_maxes[i].second&&min_maxes[i+2].second>min_maxes[i+4].second)) continue;
        double max_avg=(min_maxes[i].second+min_maxes[i+4].second)/2;
        double min_avg=(min_maxes[i+1].second+min_maxes[i+3].second)/2;
        if(!(abs(min_maxes[i].second-max_avg)<=0.015*max_avg&&abs(min_maxes[i+4].second-max_avg)<=0.015*max_avg)) continue;
        if(!(abs(min_maxes[i+1].second-min_avg)<=0.015*min_avg&&abs(min_maxes[i+3].second-min_avg)<=0.015*min_avg)) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_ihs_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    for(size_t i=((!min_maxes[0].second)?2:0);i<min_maxes.size()-4;i+=2)
    {
        if(!(min_maxes[i+2].second<min_maxes[i].second&&min_maxes[i+2].second<min_maxes[i+4].second)) continue;
        double min_avg=(min_maxes[i].second+min_maxes[i+4].second)/2;
        double max_avg=(min_maxes[i+1].second+min_maxes[i+3].second)/2;
        if(!(abs(min_maxes[i].second-min_avg)<=0.015*min_avg&&abs(min_maxes[i+4].second-min_avg)<=0.015*min_avg)) continue;
        if(!(abs(min_maxes[i+1].second-max_avg)<=0.015*max_avg&&abs(min_maxes[i+3].second-max_avg)<=0.015*max_avg)) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_btop_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    for(size_t i=1;i<min_maxes.size()-4;i+=2)
    {
        if(!(min_maxes[i].second<min_maxes[i+2].second&&min_maxes[i+2].second<min_maxes[i+4].second)) continue;
        if(!(min_maxes[i+1].second>min_maxes[i+3].second)) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_bbot_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    for(size_t i=((!min_maxes[0].second)?2:0);i<min_maxes.size()-4;i+=2)
    {
        if(!(min_maxes[i].second>min_maxes[i+2].second&&min_maxes[i+2].second>min_maxes[i+4].second)) continue;
        if(!(min_maxes[i+1].second<min_maxes[i+3].second)) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_ttop_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    for(size_t i=1;i<min_maxes.size()-4;i+=2)
    {
        if(!(min_maxes[i].second>min_maxes[i+2].second&&min_maxes[i+2].second>min_maxes[i+4].second)) continue;
        if(!(min_maxes[i+1].second<min_maxes[i+3].second)) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_tbot_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    for(size_t i=((!min_maxes[0].second)?2:0);i<min_maxes.size()-4;i+=2)
    {
        if(!(min_maxes[i].second<min_maxes[i+2].second&&min_maxes[i+2].second<min_maxes[i+4].second)) continue;
        if(!(min_maxes[i+1].second>min_maxes[i+3].second)) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_rtop_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    for(size_t i=1;i<min_maxes.size()-4;i+=2)
    {
        double max_avg=(min_maxes[i+2].second+min_maxes[i+4].second)/2;
        double min_avg=(min_maxes[i+1].second+min_maxes[i+3].second)/2;
        if(!(abs(min_maxes[i+2].second-max_avg)<=0.0075*max_avg&&abs(min_maxes[i+4].second-max_avg)<=0.0075*max_avg)) continue;
        if(!(abs(min_maxes[i+1].second-min_avg)<=0.0075*min_avg&&abs(min_maxes[i+3].second-min_avg)<=0.0075*min_avg)) continue;
        if(!(min({min_maxes[i].second,min_maxes[i+2].second,min_maxes[i+4].second})>max({min_maxes[i+1].second,min_maxes[i+3].second}))) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_rbot_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    for(size_t i=((!min_maxes[0].second)?2:0);i<min_maxes.size()-4;i+=2)
    {
        double min_avg=(min_maxes[i+2].second+min_maxes[i+4].second)/2;
        double max_avg=(min_maxes[i+1].second+min_maxes[i+3].second)/2;
        if(!(abs(min_maxes[i+2].second-min_avg)<=0.0075*min_avg&&abs(min_maxes[i+4].second-min_avg)<=0.0075*min_avg)) continue;
        if(!(abs(min_maxes[i+1].second-max_avg)<=0.0075*max_avg&&abs(min_maxes[i+3].second-max_avg)<=0.0075*max_avg)) continue;
        if(!(max({min_maxes[i].second,min_maxes[i+2].second,min_maxes[i+4].second})<min({min_maxes[i+1].second,min_maxes[i+3].second}))) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_maxes[i+4].first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_dtop_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    for(size_t i=1;i<min_maxes.size()-2;i+=2)
    {
        pair<boost::gregorian::date,double> max_next={boost::gregorian::date(),DBL_MIN};
        for(size_t j=i+2;j<min_maxes.size()-1;j+=2)
            if(min_maxes[j].second>max_next.second) max_next=min_maxes[j];
        double max_avg=(min_maxes[i].second+max_next.second)/2;
        if(!(abs(min_maxes[i].second-max_avg)<=0.015*max_avg&&abs(max_next.second-max_avg)<=0.015*max_avg)) continue;
        if(!(max_next.first-min_maxes[i].first>boost::gregorian::date_duration(22))) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,max_next.first));
    }
    return dates;
}

vector<pair<boost::gregorian::date,boost::gregorian::date> > get_dbot_models(const vector<pair<boost::gregorian::date,double> >& min_maxes)
{
    vector<pair<boost::gregorian::date,boost::gregorian::date> > dates;
    for(size_t i=((!min_maxes[0].second)?2:0);i<min_maxes.size()-2;i+=2)
    {
        pair<boost::gregorian::date,double> min_next={boost::gregorian::date(),DBL_MAX};
        for(size_t j=i+2;j<min_maxes.size()-1;j+=2)
            if(min_maxes[j].second<min_next.second) min_next=min_maxes[j];
        double min_avg=(min_maxes[i].second+min_next.second)/2;
        if(!(abs(min_maxes[i].second-min_avg)<=0.015*min_avg&&abs(min_next.second-min_avg)<=0.015*min_avg)) continue;
        if(!(min_next.first-min_maxes[i].first>boost::gregorian::date_duration(22))) continue;
        dates.emplace_back(make_pair(min_maxes[i].first,min_next.first));
    }
    return dates;
}

int main()
{
    map<string,vector<pair<boost::gregorian::date,double> > > values;
    map<string,pair<double,double> > coefs;
    std::ifstream in("torg3.txt");
    //ofstream out("results.csv");
    ifstream in_csv("results.csv");
    string trash;
    cout << fixed << setprecision(6);
    //out << fixed << setprecision(6);
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
    getline(in_csv,trash);
    while(in_csv.good())//Company;Alpha;h*
    {
        string company(""),alpha(""),h("");
        getline(in_csv,company,';');
        getline(in_csv,alpha,';');
        getline(in_csv,h,'\n');
        if((alpha=="")||(h=="")) continue;
        coefs.emplace(company,make_pair(stod(alpha),((stod(h)==0.)?.0001:stod(h))));
    }
    values.erase("");
    cout <<"Complete!\nNumber of companies: " << values.size() << '\n';
    for(auto& el:values)
    {
        vector<pair<boost::gregorian::date,double> > min_maxs_e=get_min_max_for_exp_method(el.second,(coefs.find(el.first))->second.first);
        vector<pair<boost::gregorian::date,double> > min_maxs_k=get_min_max_for_kernel_density(el.second,(coefs.find(el.first))->second.second);
        for(auto& mm:min_maxs_e) cout << mm.first << " : " << mm.second << '\n';
        cout << '\n';
        for(auto& mm:min_maxs_k) cout << mm.first << " : " << mm.second << '\n';
        cout << '\n';
    }

    //часть,нужная для заполнения res.csv. Для ее активации надо закомментить часть от values.erase("") до нее, раскомментить эту часть, а также строки, связанный с ofstring out. Время выполнения перезаписи ~1 часа, в зависимости от количества ядер процессора.
    /*
    out << "Company;Alpha;h*\n";
    for(auto& el:values)
    {
        cout << el.first << '\n';
        out << el.first << ';';
        pair<double,double> esc=get_coef_for_smoothing(el.second);
        cout << "esc: " << esc.first << " prob = " << esc.second << '\n';
        out << esc.first << ';';
        double h=get_h_for_kernel_density(el.second);
        cout << "h*: " << h << '\n';
        out << h.first << '\n';

    }*/
    delete facet;
    return 0;
}

